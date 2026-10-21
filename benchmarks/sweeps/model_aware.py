"""What the model-aware arms save on the real files: M3-Bench images scaled to a vision model's limits and the speech clips
as 16 kHz mono WAV. In-process, no server; run it with the server environment (it needs Pillow and arms.py):

    python-server/.venv/bin/python -m benchmarks.sweeps.model_aware

Visual tokens use the Claude formula, ceil(w / 28) * ceil(h / 28), after the provider's own downscale to 1568 px and
1568 tokens, so the "tokens the model sees" columns are equal by construction wherever the image exceeded the limits;
what changes is the bytes sent and the time to scale. The scaling here is Pillow's Lanczos, not the provider's.
"""

from __future__ import annotations

import csv
import io
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python-server"))
import arms  # noqa: E402
from PIL import Image  # noqa: E402


def provider_tokens(w: int, h: int) -> int:
    scale = min(1.0, 1568 / max(w, h))
    while -(-int(w * scale) // 28) * -(-int(h * scale) // 28) > 1568:
        scale *= 0.98
    return -(-int(w * scale) // 28) * -(-int(h * scale) // 28)


def main():
    out = Path("results/model_aware")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(Path("data/m3bench/images").glob("*.png")):
        data = path.read_bytes()
        w, h = Image.open(io.BytesIO(data)).size
        t = time.perf_counter()
        fitted = arms.fit_image_to_model(data)
        seconds = time.perf_counter() - t
        w2, h2 = Image.open(io.BytesIO(fitted)).size
        rows.append(["image", path.name, len(data), len(fitted), f"{w}x{h}", f"{w2}x{h2}", provider_tokens(w, h), provider_tokens(w2, h2), round(seconds, 4)])
    for path in sorted(Path("data/audio").glob("*.wav")):
        data = path.read_bytes()
        t = time.perf_counter()
        fitted = arms.audio_to_16k_mono(data)
        rows.append(["audio", path.name, len(data), len(fitted), "", "", "", "", round(time.perf_counter() - t, 4)])
    with open(out / "model_aware.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "file", "original_bytes", "arm_bytes", "original_size", "arm_size", "tokens_original", "tokens_arm", "seconds"])
        w.writerows(rows)

    lines = ["# Model-aware arms on the real files", ""]
    images = [r for r in rows if r[0] == "image"]
    changed = [r for r in images if r[2] != r[3]]
    exceeded = [r for r in images if max(map(int, r[4].split("x"))) > 1568 or r[6] != int(r[4].split("x")[0]) // 28 * 0 + r[6]]
    lines += [f"images: {len(images)}; scaled by the arm: {len(changed)} ({len(changed) / len(images):.0%}); "
              f"bytes {sum(r[2] for r in images) / 1e6:.1f} MB -> {sum(r[3] for r in images) / 1e6:.1f} MB "
              f"({sum(r[3] for r in images) / sum(r[2] for r in images):.0%} of the original)"]
    if changed:
        lines += [f"scaled images only: {sum(r[2] for r in changed) / 1e6:.1f} MB -> {sum(r[3] for r in changed) / 1e6:.1f} MB; "
                  f"median {st.median(r[3] / r[2] for r in changed):.0%} of the original; median time {st.median(r[8] for r in changed) * 1000:.0f} ms; "
                  f"tokens the model sees (original with provider downscale -> arm): "
                  f"{st.mean(r[6] for r in changed):.0f} -> {st.mean(r[7] for r in changed):.0f}"]
    big = [r for r in images if max(map(int, r[4].split("x"))) > 1568]
    skipped = [r for r in big if r[2] == r[3]]
    lines += [f"images over the 1568 px limit: {len(big)}; of those, the arm kept the original (scaling would not shrink the file) for {len(skipped)}"]
    audio = [r for r in rows if r[0] == "audio"]
    lines += [f"audio clips: {len(audio)}; bytes {sum(r[2] for r in audio) / 1e6:.1f} MB -> {sum(r[3] for r in audio) / 1e6:.1f} MB "
              f"({sum(r[3] for r in audio) / sum(r[2] for r in audio):.0%}); median time {st.median(r[8] for r in audio) * 1000:.0f} ms (ffmpeg start-up included)"]
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
