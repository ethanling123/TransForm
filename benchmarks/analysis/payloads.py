"""Summarize the unstructured payload runs (results/payloads/): fixed arms vs the selector, per payload kind."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

R = Path("results/payloads")
RUNS = [("localhost", "localhost (no emulation)"), ("lan", "LAN (1 ms, 1 Gbit)"), ("wan", "WAN (40 ms, 50 Mbit)"),
        ("cellular", "Cellular (80 ms, 10 Mbit, 0.1% loss)"), ("badwifi", "BadWifi (30 ms, 5 Mbit, 1% loss)")]
STRATEGIES = ["raw", "gzip", "zstd", "brotli", "selector min_bytes", "selector min_latency"]


def load(name):
    with open(R / f"{name}.csv") as f:
        return list(csv.DictReader(f))


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(out + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def summary(run):
    rows = load(run)
    out = []
    for kind in ("image", "audio", "text"):
        rs = [r for r in rows if r["kind"] == kind]
        payloads = {r["payload"] for r in rs}
        raw = sum(int(r["raw_bytes"]) for r in rs if r["strategy"] == "raw")
        for strategy in STRATEGIES:
            s = [r for r in rs if r["strategy"] == strategy]
            wire = sum(int(r["wire_bytes"]) for r in s)
            latency = sum(float(r["latency_s"]) for r in s)
            chosen = ""
            if strategy.startswith("selector"):
                chosen = ", ".join(f"{k.replace('_blob', '')}: {v}" for k, v in sorted(Counter(r["arm"] for r in s).items()))
            out.append([f"{kind} ({len(payloads)}, {raw / 1e6:.0f} MB)", strategy, f"{wire / 1e6:.2f}",
                        f"{wire / raw * 100:.0f}%", f"{latency:.1f}", chosen])
    return table(["payloads", "strategy", "wire MB", "% of raw", "total latency (s)", "selector choices"], out), rows


def mismatches(run):
    return sum(r["ok"] != "True" for r in load(run))


def main():
    for run, label in RUNS:
        if not (R / f"{run}.csv").exists():
            continue
        text, _ = summary(run)
        print(f"#### {label}\n\n{text}\n\nRound-trip mismatches: {mismatches(run)}\n")


if __name__ == "__main__":
    main()
