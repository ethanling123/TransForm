"""Image and audio codec sweep: bytes, encode time, decode time (and image fidelity) for every arm.

In-process, no server. Images are the M3-Bench images (data/m3bench/images); audio is the synthesized speech set
(data/audio, see benchmarks/datasets/speech.py). Audio needs ffmpeg on the PATH.

    .venv/bin/python -m benchmarks.sweeps.media --workers 4

Image fidelity is PSNR against the original (alpha flattened onto white): a proxy for how much a lossy arm changes
the pixels, not a measure of whether a model still answers correctly. Audio fidelity is not measured here.
Timings with --workers > 1 are inflated by contention; use --workers 1 for clean timings.
"""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_ARMS = [
    # name, format, save kwargs, max edge (None = full size), lossless
    ("png", "PNG", {}, None, True),
    ("webp-lossless", "WEBP", {"lossless": True, "method": 4}, None, True),
    ("webp-q90", "WEBP", {"quality": 90}, None, False),
    ("webp-q80", "WEBP", {"quality": 80}, None, False),
    ("webp-q60", "WEBP", {"quality": 60}, None, False),
    ("avif-q80", "AVIF", {"quality": 80, "speed": 6}, None, False),
    ("avif-q60", "AVIF", {"quality": 60, "speed": 6}, None, False),
    ("avif-q40", "AVIF", {"quality": 40, "speed": 6}, None, False),
    ("jpeg-q90", "JPEG", {"quality": 90}, None, False),
    ("jpeg-q75", "JPEG", {"quality": 75}, None, False),
    ("png-1024px", "PNG", {}, 1024, False),
    ("webp-q80-1024px", "WEBP", {"quality": 80}, 1024, False),
]

AUDIO_ARMS = [
    # name, extension, ffmpeg output args (input is 44.1 kHz mono 16-bit WAV), lossless
    ("wav-44k", "wav", ["-c:a", "pcm_s16le"], True),
    ("wav-16k", "wav", ["-ar", "16000", "-c:a", "pcm_s16le"], False),
    ("flac-44k", "flac", ["-c:a", "flac"], True),
    ("flac-16k", "flac", ["-ar", "16000", "-c:a", "flac"], False),
    ("opus-32k", "opus", ["-ar", "16000", "-c:a", "libopus", "-b:a", "32k"], False),
    ("opus-24k", "opus", ["-ar", "16000", "-c:a", "libopus", "-b:a", "24k"], False),
    ("opus-12k", "opus", ["-ar", "16000", "-c:a", "libopus", "-b:a", "12k"], False),
    ("aac-48k", "m4a", ["-c:a", "aac", "-b:a", "48k"], False),
    ("mp3-64k", "mp3", ["-c:a", "libmp3lame", "-b:a", "64k"], False),
]


def to_rgb(im: Image.Image) -> Image.Image:
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        flat = Image.new("RGB", im.size, (255, 255, 255))
        flat.paste(im, mask=im.getchannel("A"))
        return flat
    return im.convert("RGB")


def psnr(a: Image.Image, b: Image.Image) -> float:
    x, y = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    mse = float(np.mean((x - y) ** 2))
    return 99.0 if mse == 0 else float(10 * np.log10(255.0**2 / mse))


def image_rows(path: str):
    original = Image.open(path)
    original.load()
    raw_bytes = Path(path).stat().st_size
    reference = to_rgb(original)
    rows = []
    for name, fmt, kwargs, max_edge, lossless in IMAGE_ARMS:
        if fmt == "PNG":
            source = original
        elif fmt == "WEBP":  # keeps alpha; WebP has no palette mode
            source = original if original.mode in ("RGB", "RGBA") else original.convert("RGBA")
        else:  # JPEG has no alpha; AVIF is run on the flattened image too
            source = reference
        if max_edge:
            source = source.copy()
            source.thumbnail((max_edge, max_edge), Image.LANCZOS)
        buf = io.BytesIO()
        t = time.perf_counter()
        source.save(buf, fmt, **kwargs)
        encode_s = time.perf_counter() - t
        data = buf.getvalue()
        t = time.perf_counter()
        decoded = Image.open(io.BytesIO(data))
        decoded.load()
        decode_s = time.perf_counter() - t
        same_size = decoded.size == reference.size
        fidelity = psnr(reference, to_rgb(decoded)) if same_size else ""
        rows.append([Path(path).name, original.width, original.height, original.mode, raw_bytes, name,
                     len(data), encode_s, decode_s, fidelity])
    return rows


def ffmpeg(args):
    t = time.perf_counter()
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)
    return time.perf_counter() - t


def audio_rows(path: str):
    rows = []
    duration = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                                    capture_output=True, text=True, check=True).stdout)
    raw_bytes = Path(path).stat().st_size
    with tempfile.TemporaryDirectory() as tmp:
        for name, ext, args, _ in AUDIO_ARMS:
            out = Path(tmp) / f"x.{ext}"
            encode_s = ffmpeg(["-i", path, "-ac", "1", *args, str(out)])
            decode_s = ffmpeg(["-i", str(out), "-f", "null", "-"])
            rows.append([Path(path).name, duration, raw_bytes, name, out.stat().st_size, encode_s, decode_s])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/m3bench/images")
    parser.add_argument("--audio", default="data/audio")
    parser.add_argument("--stride", type=int, default=1, help="use every Nth image")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", default="results/media")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    images = sorted(Path(args.images).glob("*.png"))[:: args.stride]
    audio = sorted(Path(args.audio).glob("*.wav"))

    with ProcessPoolExecutor(args.workers) as pool:
        image_results = list(pool.map(image_rows, map(str, images)))
        print(f"images: {len(images)} done", flush=True)
        audio_results = list(pool.map(audio_rows, map(str, audio)))
        print(f"audio: {len(audio)} done", flush=True)

    with open(out / "images.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "width", "height", "mode", "png_bytes", "arm", "bytes", "encode_s", "decode_s", "psnr_db"])
        w.writerows(row for rows in image_results for row in rows)
    with open(out / "audio.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "duration_s", "wav_bytes", "arm", "bytes", "encode_s", "decode_s"])
        w.writerows(row for rows in audio_results for row in rows)


if __name__ == "__main__":
    main()
