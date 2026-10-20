"""Is a lossy arm worth it? Time to get the payload to the model (encode + transfer + decode) against the lossless
original, per link, next to what the arm did to the model's answers.

Encode and decode times are the in-process measurements from results/media (benchmarks.sweeps.media; use --workers 1
there for timings without contention); transfer is
bytes / link speed with no round-trip term (it is the same for every arm). Accuracy comes from results/lossy.

    .venv/bin/python -m benchmarks.analysis.lossy_worth
"""

from __future__ import annotations

import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

LINKS = [("LAN", 1000.0), ("WAN", 50.0), ("cellular", 10.0), ("bad wifi", 5.0)]
MEDIA = Path("results/media")


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def by_item(rows, key):
    out = defaultdict(dict)
    for r in rows:
        out[r[key]][r["arm"]] = r
    return out


def total(row, bytes_col, mbps):
    return float(row["encode_s"]) + int(row[bytes_col]) * 8 / (mbps * 1e6) + float(row["decode_s"])


def speedups(items, base_arm, orig_col, bytes_col, mbps):
    """Per arm: median speed-up over the base arm, and the share of items it is faster on."""
    out = defaultdict(list)
    for arms in items.values():
        base = arms[base_arm]
        base_t = float(base["decode_s"]) + int(base[orig_col]) * 8 / (mbps * 1e6)  # the original file as stored: no re-encode
        for name, r in arms.items():
            out[name].append(base_t / total(r, bytes_col, mbps))
    return {a: (st.median(v), sum(x > 1 for x in v) / len(v)) for a, v in out.items()}


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(out + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def accuracy_images():
    path = Path("results/lossy/image_accuracy_vertex.csv")
    if not path.exists():
        return {}
    by = defaultdict(list)
    for r in load(path):
        by[r["arm"]].append(float(r["objects_f1"]))
    return {a: st.mean(v) for a, v in by.items()}


def accuracy_audio():
    by = defaultdict(list)
    path = Path("results/lossy/audio_accuracy.csv")
    if not path.exists():
        return {}
    rows = load(path)
    base = {r["clip"]: float(r["wer"]) for r in rows if r["arm"] == "original"}
    for r in rows:
        by[r["arm"]].append(100 * (float(r["wer"]) - base[r["clip"]]))
    return {a: st.mean(v) for a, v in by.items()}


def main():
    images = by_item(load(MEDIA / "images.csv"), "image")
    audio = by_item(load(MEDIA / "audio.csv"), "clip")
    out = ["# Is a lossy arm worth it?", "",
           "Speed-up = time to deliver the lossless original / time to encode, send and decode the arm "
           "(median over items; above 1x the arm is faster). `faster on` is the share of items where it is.", ""]
    for title, items, base, orig_col, bytes_col, acc, acc_name, fmt in (
            ("Images (PNG original)", images, "png", "png_bytes", "bytes", accuracy_images(), "objects agreement (Gemini)", "{:.2f}"),
            ("Audio (WAV original)", audio, "wav-44k", "wav_bytes", "bytes", accuracy_audio(), "WER change vs original (pp)", "{:+.2f}")):
        names = list(next(iter(items.values())))
        ratios = {a: [int(items[i][a]["bytes"]) / int(items[i][base][orig_col]) for i in items] for a in names}
        header = ["arm", "bytes vs original (median; smaller on)", acc_name] + [f"{n} speed-up (faster on)" for n, _ in LINKS]
        per_link = {n: speedups(items, base, orig_col, bytes_col, m) for n, m in LINKS}
        rows = []
        for a in names:
            cells = [f"{per_link[n][a][0]:.2f}x ({per_link[n][a][1]:.0%})" for n, _ in LINKS]
            rows.append([a, f"{st.median(ratios[a]):.1%}; {sum(x < 1 for x in ratios[a]) / len(ratios[a]):.0%}", fmt.format(acc[a]) if a in acc else "-", *cells])
        out += [f"## {title}", "", table(header, rows), ""]
    text = "\n".join(out)
    Path("results/lossy/worth.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
