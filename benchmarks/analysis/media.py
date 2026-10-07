"""Summarize the image and audio codec sweep (results/media/)."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

R = Path("results/media")


def load(name):
    with open(R / name) as f:
        return list(csv.DictReader(f))


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(out + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def images():
    rows = load("images.csv")
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    n_images = len({r["image"] for r in rows})
    source_total = sum(int(r["png_bytes"]) for r in by_arm["png"])
    out = []
    for arm, rs in by_arm.items():
        total = sum(int(r["bytes"]) for r in rs)
        psnr = [float(r["psnr_db"]) for r in rs if r["psnr_db"] != ""]
        if arm.endswith("px"):  # PSNR is only defined for images the resize left unchanged
            fidelity = f"n/a: resized for {len(rs) - len(psnr)} of {len(rs)} images"
        elif arm in ("png", "webp-lossless"):
            fidelity = "lossless"
        else:
            fidelity = f"{statistics.median(psnr):.1f} (min {min(psnr):.1f})"
        out.append([arm, f"{total / 1e6:.1f}", f"{total / source_total * 100:.1f}%",
                    f"{statistics.median(float(r['encode_s']) for r in rs) * 1e3:.0f}",
                    f"{statistics.median(float(r['decode_s']) for r in rs) * 1e3:.0f}", fidelity])
    head = ["arm", "total MB", "% of source PNG", "median encode ms", "median decode ms", "PSNR dB, median (min)"]
    return n_images, source_total, table(head, out)


def audio():
    rows = load("audio.csv")
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    wav_total = sum(int(r["wav_bytes"]) for r in by_arm["wav-44k"])
    seconds = sum(float(r["duration_s"]) for r in by_arm["wav-44k"])
    out = []
    for arm, rs in by_arm.items():
        total = sum(int(r["bytes"]) for r in rs)
        out.append([arm, f"{total / 1e6:.2f}", f"{total / wav_total * 100:.1f}%", f"{total * 8 / seconds / 1e3:.0f}",
                    f"{statistics.median(float(r['encode_s']) for r in rs) * 1e3:.0f}",
                    f"{statistics.median(float(r['decode_s']) for r in rs) * 1e3:.0f}"])
    head = ["arm", "total MB", "% of 44.1 kHz WAV", "avg kbit/s", "median encode ms", "median decode ms"]
    return len(by_arm["wav-44k"]), seconds, table(head, out)


def main():
    n, source, imgs = images()
    print(f"#### Images: {n} M3-Bench PNGs ({source / 1e6:.1f} MB)\n\n{imgs}\n")
    clips, seconds, aud = audio()
    print(f"#### Audio: {clips} synthesized speech clips ({seconds:.0f} s)\n\n{aud}")


if __name__ == "__main__":
    main()
