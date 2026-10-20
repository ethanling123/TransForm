"""Summarize the lossy-arm accuracy runs (results/lossy/): audio word error rate and image answer agreement.

    .venv/bin/python -m benchmarks.analysis.lossy
"""

from __future__ import annotations

import csv
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

R = Path("results/lossy")


def load(name):
    with open(R / name) as f:
        return list(csv.DictReader(f))


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(out + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def ci(values, reps=2000):
    """95% bootstrap interval for the mean."""
    rng = random.Random(0)
    means = sorted(st.mean(rng.choices(values, k=len(values))) for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps)]


def by_arm(rows):
    out = defaultdict(list)
    for r in rows:
        out[r["arm"]].append(r)
    return out


def audio_noise():
    """Change in mean WER (percentage points) against the same noisy clip left unencoded, per signal-to-noise ratio."""
    levels = [("clean", "audio_accuracy.csv")] + [(f"{snr} dB", f"audio_accuracy_snr{snr}.csv") for snr in (20, 10, 5)
                                                  if (R / f"audio_accuracy_snr{snr}.csv").exists()]
    data = {}
    for name, filename in levels:
        groups = by_arm(load(filename))
        base = {r["clip"]: float(r["wer"]) for r in groups["original"]}
        data[name] = {arm: (st.mean(100 * (float(r["wer"]) - base[r["clip"]]) for r in rs), 100 * st.mean(float(r["wer"]) for r in groups["original"]))
                      for arm, rs in groups.items()}
    arms = list(data["clean"])
    clean = by_arm(load("audio_accuracy.csv"))
    header = ["audio arm", "bytes (% of WAV)"] + [f"WER change, {n}" for n, _ in levels]
    rows = [[a, f"{100 * st.mean(int(r['bytes']) / int(r['wav_bytes']) for r in clean[a]):.1f}%"] + [f"{data[n][a][0]:+.2f} pp" for n, _ in levels] for a in arms]
    floor = ["unencoded WER", ""] + [f"{data[n]['original'][1]:.2f}%" for n, _ in levels]
    return table(header, rows + [floor])


def audio():
    rows = load("audio_accuracy.csv")
    groups = by_arm(rows)
    base = {r["clip"]: float(r["wer"]) for r in groups["original"]}
    out = []
    for arm, rs in groups.items():
        pct = st.mean(int(r["bytes"]) / int(r["wav_bytes"]) for r in rs) * 100
        wers = [float(r["wer"]) for r in rs]
        worse = sum(float(r["wer"]) > base[r["clip"]] + 1e-9 for r in rs)
        delta = [100 * (float(r["wer"]) - base[r["clip"]]) for r in rs]
        out.append([arm, f"{pct:.1f}%", f"{100 * st.mean(wers):.2f}%", f"{st.mean(delta):+.2f} pp", f"{max(delta):+.1f} pp",
                    f"{worse} of {len(rs)}"])
    clips = len(base)
    words = "ground truth is the exact script each clip was synthesized from"
    return clips, table(["audio arm", "bytes (% of WAV)", "mean WER", "change vs original", "worst clip", "clips worse"], out), words


def images(filename="image_accuracy.csv"):
    rows = load(filename)
    groups = by_arm(rows)
    out = []
    for arm, rs in groups.items():
        if len(rs) < 5:
            continue
        ratios = [int(r["bytes"]) / int(r["png_bytes"]) for r in rs]
        pct = st.median(ratios) * 100
        smaller = sum(x < 1 for x in ratios) / len(ratios)
        psnr = [float(r["psnr_db"]) for r in rs if r["psnr_db"] not in ("", None)]
        ssim = [float(r["ssim"]) for r in rs if r["ssim"] not in ("", None)]
        ocr = [float(r["ocr_f1"]) for r in rs]
        obj = [float(r["objects_f1"]) for r in rs]
        lo_o, hi_o = ci(ocr)
        lo_b, hi_b = ci(obj)
        out.append([arm, len(rs), f"{pct:.1f}% ({smaller:.0%} smaller)", f"{st.mean(psnr):.1f}" if psnr else "n/a (resized)",
                    f"{st.mean(ssim):.3f}" if ssim else "n/a (resized)", f"{st.mean(ocr):.2f} ({lo_o:.2f}-{hi_o:.2f})",
                    f"{st.mean(obj):.2f} ({lo_b:.2f}-{hi_b:.2f})"])
    out.sort(key=lambda r: -float(r[2].split("%")[0]))
    return len({r["image"] for r in rows}), table(
        ["image arm", "n", "bytes, median % of PNG (share of images smaller)", "PSNR dB", "SSIM", "OCR agreement (95% CI)", "objects agreement (95% CI)"], out)


def documents(filename):
    rows = load(filename)
    groups = by_arm(rows)
    out = []
    for arm, rs in groups.items():
        cer = [float(r["cer"]) for r in rs]
        wrong = sum(c > 0 for c in cer)
        out.append([arm, len(rs), f"{st.mean(int(r['bytes']) / int(r['png_bytes']) for r in rs) * 100:.0f}%", f"{100 * st.mean(cer):.2f}%",
                    f"{100 * max(cer):.1f}%", f"{wrong} of {len(rs)}"])
    out.sort(key=lambda r: (float(r[3].rstrip("%")), r[0]))
    fonts = sorted({int(r["font_px"]) for r in rows})
    return fonts, len({r["document"] for r in rows}), table(["arm", "n", "bytes (% of PNG)", "mean CER", "worst CER", "images with any error"], out)


def main():
    clips, audio_table, _ = audio()
    out = f"""#### Audio: word error rate with a local Whisper model ({clips} clips)

{audio_table}

#### Audio under added noise: change in word error rate against the unencoded noisy clip

{audio_noise()}"""
    for filename, label in (("document_accuracy.csv", "documents"), ("document_accuracy_harsh.csv", "documents, harsher settings")):
        if (R / filename).exists():
            fonts, n_docs, doc_table = documents(filename)
            out += f"""

#### Rendered documents ({label}): character error rate of a local 7B vision model against the true text ({n_docs} texts at {", ".join(map(str, fonts))} px)

{doc_table}"""
    for filename, label in (("image_accuracy_vertex.csv", "Gemini 2.5 Flash on Vertex AI"), ("image_accuracy.csv", "local 7B vision model")):
        if (R / filename).exists():
            n_images, image_table = images(filename)
            out += f"""

#### Images, {label}: agreement with its answer on the untouched image ({n_images} images)

{image_table}"""
    print(out)


if __name__ == "__main__":
    main()
