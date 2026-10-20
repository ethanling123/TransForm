"""Does a lossy arm change what a model gets out of the payload? Local, no API key.

audio   Each speech clip goes through every audio arm (encode, then decode to the 16 kHz mono a speech model takes)
        and is transcribed with a local Whisper model. Word error rate is measured against the exact script the
        clip was synthesized from (data/audio_ref/*.txt), so it is correctness, not agreement.
images  Each M3-Bench image goes through the image arms and a local vision model (Ollama) is asked two fixed
        questions about the decoded pixels: transcribe the visible text, and list the main objects. There is no
        ground truth for those, so the score is agreement with the model's own answer on the untouched image
        (token F1). That measures whether the arm changed what the model sees, not whether the model is right.

documents  Synthetic document images (known text rendered at several font sizes). Unlike the natural images
        these have an exact ground truth, so the score is character and word error rate of the model's transcription,
        and a lossy arm is judged on correctness, not on agreement with the model.

Every image run includes a `png-rerun` control: the untouched pixels asked a second time, so the gap between two
identical inputs (the model's own run-to-run noise) is measured, not assumed. `audio --snr N` mixes white noise in at
N dB signal-to-noise ratio before the clip is encoded.

Rows are appended as they finish and finished (item, arm) pairs are skipped, so a run can be stopped and resumed.

    .venv/bin/python -m benchmarks.sweeps.lossy audio
    .venv/bin/python -m benchmarks.sweeps.lossy images --limit 30
    .venv/bin/python -m benchmarks.sweeps.lossy documents
    .venv/bin/python -m benchmarks.sweeps.lossy documents --harsh --fonts 7 9
    .venv/bin/python -m benchmarks.sweeps.lossy audio --snr 10
    VERTEX_API_KEY=... .venv/bin/python -m benchmarks.sweeps.lossy images --backend vertex
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import re
import subprocess
import tempfile
import textwrap
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import jiwer
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity

from benchmarks.sweeps.media import AUDIO_ARMS, IMAGE_ARMS, psnr, to_rgb

WHISPER_MODEL = "mlx-community/whisper-small.en-mlx"
VISION_MODEL = "qwen2.5vl:7b"
OLLAMA = "http://localhost:11434"
DOC_PROMPT = "Transcribe all the text in this image exactly as written. Output only the text."
PROMPTS = {
    "ocr": "Transcribe every piece of text visible in this image, one item per line. If there is no text, answer NONE.",
    "objects": "List the main objects or items visible in this image as a comma-separated list of short noun phrases "
               "(at most 12). Answer with the list only.",
}
# Near-lossless control: visually identical, so any change in the model's answer here is the model's own sensitivity.
NOISE_FLOOR_ARM = ("jpeg-q95", "JPEG", {"quality": 95}, None, False)
RERUN_ARM = ("png-rerun", "PNG", {}, None, True)
HALF_SIZE_ARM = ("png-0.5x", "PNG", {}, 0.5, False)
HARSH_ARMS = [("jpeg-q30", "JPEG", {"quality": 30}, None, False), ("webp-q30", "WEBP", {"quality": 30}, None, False),
              ("avif-q20", "AVIF", {"quality": 20, "speed": 6}, None, False), ("png-0.25x", "PNG", {}, 0.25, False)]
WER_NORMALIZE = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip(),
                               jiwer.ReduceToListOfListOfWords()])


def done_keys(path: Path, key_cols: int) -> set:
    if not path.exists():
        return set()
    with open(path) as f:
        return {tuple(row[:key_cols]) for row in list(csv.reader(f))[1:]}


def append(path: Path, header: list, row: list) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def transcribe(path: str) -> str:
    import mlx_whisper
    return mlx_whisper.transcribe(path, path_or_hf_repo=WHISPER_MODEL, language="en")["text"]


def wer(reference: str, hypothesis: str) -> float:
    return jiwer.wer(reference, hypothesis, reference_transform=WER_NORMALIZE, hypothesis_transform=WER_NORMALIZE)


def add_noise(src: Path, dst: Path, snr_db: float) -> None:
    """White noise at the given signal-to-noise ratio, fixed seed, written as 16-bit WAV with the clip's own format."""
    with wave.open(str(src)) as w:
        params = w.getparams()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    noise = np.random.default_rng(0).standard_normal(len(x)).astype(np.float32)
    noise *= np.sqrt(np.mean(x**2) / 10 ** (snr_db / 10)) / np.sqrt(np.mean(noise**2))
    with wave.open(str(dst), "wb") as w:
        w.setparams(params)
        w.writeframes(np.clip(x + noise, -32768, 32767).astype(np.int16).tobytes())


def run_audio(args) -> None:
    out = Path(args.out_dir) / ("audio_accuracy.csv" if args.snr is None else f"audio_accuracy_snr{args.snr:g}.csv")
    header = ["clip", "arm", "bytes", "wav_bytes", "wer", "wer_vs_original_transcript"]
    done = done_keys(out, 2)
    clips = sorted(Path(args.audio).glob("*.wav"))
    for clip in clips:
        reference = clip.with_suffix(".txt").read_text()
        with tempfile.TemporaryDirectory() as tmp:
            if args.snr is not None:
                noisy = Path(tmp) / clip.name
                add_noise(clip, noisy, args.snr)
                clip_input = noisy
            else:
                clip_input = clip
            decoded = Path(tmp) / "decoded.wav"
            ffmpeg("-i", str(clip_input), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(decoded))
            original_text = transcribe(str(decoded))
            if (clip.name, "original") not in done:
                append(out, header, [clip.name, "original", clip.stat().st_size, clip.stat().st_size,
                                     wer(reference, original_text), 0.0])
            for name, ext, ffargs, _ in AUDIO_ARMS:
                if (clip.name, name) in done:
                    continue
                encoded = Path(tmp) / f"x.{ext}"
                ffmpeg("-i", str(clip_input), "-ac", "1", *ffargs, str(encoded))
                ffmpeg("-i", str(encoded), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(decoded))
                text = transcribe(str(decoded))
                append(out, header, [clip.name, name, encoded.stat().st_size, clip.stat().st_size, wer(reference, text),
                                     wer(original_text, text)])
        print(f"{clip.name} done", flush=True)


def png_b64(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ask(client: httpx.Client, image_b64: str, prompt: str) -> str:
    for attempt in range(4):
        resp = client.post(f"{OLLAMA}/api/chat", json={
            "model": VISION_MODEL, "stream": False, "options": {"temperature": 0, "seed": 0, "num_predict": 250},
            "messages": [{"role": "user", "content": prompt, "images": [image_b64]}]})
        if resp.status_code != 500:  # Ollama returns 500 now and then while it reloads the model
            break
        time.sleep(5 * (attempt + 1))
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def ask_vertex(client: httpx.Client, image_b64: str, prompt: str) -> str:
    """Same question to a Gemini model on Vertex AI in express mode (key in VERTEX_API_KEY or GEMINI_API_KEY)."""
    key = os.environ.get("VERTEX_API_KEY") or os.environ["GEMINI_API_KEY"]
    model = os.environ.get("VERTEX_MODEL", "gemini-2.5-flash")
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/png", "data": image_b64}}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 400, "thinkingConfig": {"thinkingBudget": 0}}}
    for attempt in range(6):
        try:
            resp = client.post(f"https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:generateContent",
                               json=body, params={"key": key})
        except httpx.TransportError:  # a dropped connection under parallel load; try again
            time.sleep(2 ** attempt)
            continue
        if resp.status_code in (429, 500, 503):
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        parts = (resp.json().get("candidates") or [{}])[0].get("content", {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts)
    raise RuntimeError("Vertex request kept failing after 6 attempts")


def tokens(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(reference: str, answer: str) -> float:
    """F1 over the sets of words in the two answers. Sets, not counts: a model that lists the same label ten times
    in one run and twelve in another has not changed its answer."""
    ref, ans = set(tokens(reference)), set(tokens(answer))
    if not ref and not ans:
        return 1.0
    common = len(ref & ans)
    if common == 0:
        return 0.0
    p, r = common / len(ans), common / len(ref)
    return 2 * p * r / (p + r)


def apply_arm(original: Image.Image, reference: Image.Image, spec):
    """Encode and decode one image through an arm, returning (bytes, decoded image)."""
    _, fmt, kwargs, max_edge, _ = spec
    if fmt == "PNG":
        source = original
    elif fmt == "WEBP":
        source = original if original.mode in ("RGB", "RGBA") else original.convert("RGBA")
    else:
        source = reference
    # Always a private copy: Pillow keeps the save options on the image object, so threads saving the same
    # image with different options would overwrite each other's (lossy WebP came out lossless in a parallel run).
    source = source.copy()
    if max_edge:
        edge = max_edge if max_edge > 1 else max(source.size) * max_edge
        source.thumbnail((edge, edge), Image.LANCZOS)
    buf = io.BytesIO()
    source.save(buf, fmt, **kwargs)
    decoded = Image.open(io.BytesIO(buf.getvalue()))
    decoded.load()
    return len(buf.getvalue()), decoded


def run_images(args) -> None:
    out = Path(args.out_dir) / ("image_accuracy.csv" if args.backend == "ollama" else f"image_accuracy_{args.backend}.csv")
    vision = ask if args.backend == "ollama" else ask_vertex
    header = ["image", "arm", "bytes", "png_bytes", "psnr_db", "ssim", "ocr_f1", "objects_f1", "seconds"]
    done = done_keys(out, 2)
    wanted = set(args.arms.split(",")) if args.arms else None
    specs = [s for s in [*IMAGE_ARMS, NOISE_FLOOR_ARM, RERUN_ARM] if s[0] != "png" and (wanted is None or s[0] in wanted)]
    images = sorted(Path(args.images).glob("*.png"))[:: args.stride][: args.limit]
    with httpx.Client(timeout=600) as client:
        for path in images:
            original = Image.open(path)
            original.load()
            reference = to_rgb(original)
            todo = [spec for spec in specs if (path.name, spec[0]) not in done]
            if not todo:
                continue
            baseline = {k: vision(client, png_b64(reference), p) for k, p in PROMPTS.items()}

            def measure(spec):
                t = time.perf_counter()
                size, decoded = apply_arm(original, reference, spec)
                shown = to_rgb(decoded)
                same_size = shown.size == reference.size
                p = psnr(reference, shown) if same_size else ""
                s = structural_similarity(np.asarray(reference), np.asarray(shown), channel_axis=2) if same_size else ""
                answers = {k: vision(client, png_b64(shown), pr) for k, pr in PROMPTS.items()}
                row = [path.name, spec[0], size, path.stat().st_size, p, s, token_f1(baseline["ocr"], answers["ocr"]),
                       token_f1(baseline["objects"], answers["objects"]), time.perf_counter() - t]
                return row, {"image": path.name, "arm": spec[0], "answers": answers, "baseline": baseline}

            # a hosted model takes several requests at once; the local one is one GPU, so it runs one at a time
            with ThreadPoolExecutor(args.workers if args.backend != "ollama" else 1) as pool:
                for row, log_entry in pool.map(measure, todo):
                    with open(out.with_name(out.stem.replace("accuracy", "answers") + ".jsonl"), "a") as log:
                        log.write(json.dumps(log_entry) + "\n")
                    append(out, header, row)
            print(f"{path.name} done", flush=True)


def render_document(text: str, font_px: int, width: int = 900) -> tuple[Image.Image, str]:
    """White page with the text wrapped in black; returns the image and the exact wrapped text it contains."""
    font = ImageFont.load_default(size=font_px)
    chars = max(20, int(width / (font_px * 0.55)))
    lines = textwrap.wrap(text, chars)
    line_h = int(font_px * 1.4)
    im = Image.new("RGB", (width, line_h * len(lines) + 40), "white")
    draw = ImageDraw.Draw(im)
    for i, line in enumerate(lines):
        draw.text((20, 20 + i * line_h), line, fill="black", font=font)
    return im, " ".join(lines)


def run_documents(args) -> None:
    out = Path(args.out_dir) / ("document_accuracy_harsh.csv" if args.harsh else "document_accuracy.csv")
    header = ["document", "font_px", "arm", "bytes", "png_bytes", "cer", "wer", "seconds"]
    done = done_keys(out, 3)
    passages = [l.strip() for l in (Path(__file__).parents[1] / "datasets" / "speech_passages.txt").read_text().splitlines() if l.strip()]
    specs = [s for s in [*IMAGE_ARMS, NOISE_FLOOR_ARM, RERUN_ARM, HALF_SIZE_ARM] if s[0] != "png" and s[3] != 1024]
    if args.harsh:
        specs = [RERUN_ARM, *HARSH_ARMS]
    with httpx.Client(timeout=600) as client:
        for i in range(len(passages)):
            passage = " ".join(passages[(i + k) % len(passages)] for k in range(3))
            for font_px in args.fonts:
                name = f"doc{i:02d}"
                original, reference = render_document(passage, font_px)
                buf = io.BytesIO()
                original.save(buf, "PNG")
                png_bytes = len(buf.getvalue())
                for spec in [("png", "PNG", {}, None, True), *specs]:
                    if (name, str(font_px), spec[0]) in done:
                        continue
                    t = time.perf_counter()
                    size, decoded = apply_arm(original, original, spec)
                    try:
                        answer = ask(client, png_b64(decoded), DOC_PROMPT)
                    except httpx.HTTPStatusError:  # the model refuses images below its minimum size (seen at quarter size); skipped, not scored
                        print(f"{name}@{font_px} {spec[0]}: model rejected a {decoded.size[0]}x{decoded.size[1]} image", flush=True)
                        continue
                    append(out, header, [name, font_px, spec[0], size, png_bytes, text_error(jiwer.cer, reference, answer),
                                         text_error(jiwer.wer, reference, answer), time.perf_counter() - t])
                print(f"{name}@{font_px} done", flush=True)


def text_error(metric, reference: str, answer: str) -> float:
    return metric(reference, answer, **({"reference_transform": WER_NORMALIZE, "hypothesis_transform": WER_NORMALIZE}
                                        if metric is jiwer.wer else {}))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="kind", required=True)
    audio = sub.add_parser("audio")
    audio.add_argument("--audio", default="data/audio_ref")
    images = sub.add_parser("images")
    images.add_argument("--images", default="data/m3bench/images")
    images.add_argument("--stride", type=int, default=7)
    images.add_argument("--limit", type=int, default=30)
    images.add_argument("--backend", choices=["ollama", "vertex"], default="ollama",
                        help="ollama: local vision model; vertex: Gemini on Vertex AI express mode (key in VERTEX_API_KEY or GEMINI_API_KEY)")
    images.add_argument("--workers", type=int, default=4, help="concurrent requests for a hosted backend")
    images.add_argument("--arms", default="", help="comma-separated arm names (default: every non-PNG image arm)")
    audio.add_argument("--snr", type=float, default=None, help="mix in white noise at this signal-to-noise ratio (dB)")
    documents = sub.add_parser("documents")
    documents.add_argument("--fonts", type=int, nargs="+", default=[11, 14, 18], help="font sizes in pixels")
    documents.add_argument("--harsh", action="store_true", help="only the harshest settings (JPEG/WebP q30, AVIF q20, quarter size)")
    for p in (audio, images, documents):
        p.add_argument("--out-dir", default="results/lossy")
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.kind == "images" and args.backend == "vertex" and not (os.environ.get("VERTEX_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        raise SystemExit("VERTEX_API_KEY (or GEMINI_API_KEY) is not set")
    {"audio": run_audio, "images": run_images, "documents": run_documents}[args.kind](args)


if __name__ == "__main__":
    main()
