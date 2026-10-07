"""Summarize the Python-server network-emulation runs (results/network/) into markdown tables."""

from __future__ import annotations

import csv
from pathlib import Path

RESULTS = Path("results")
PROFILES = [("lan", "LAN (1 ms, 1 Gbit)"), ("wan", "WAN (40 ms, 50 Mbit)"),
            ("cellular", "Cellular (80 ms, 10 Mbit, 0.1% loss)"), ("badwifi", "BadWifi (30 ms, 5 Mbit, 1% loss)")]
STRATEGIES = [("json", "always JSON"), ("parquet", "always Parquet"), ("arrow_ipc", "always Arrow IPC")]


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def p95(values):
    values = sorted(values)
    return values[min(len(values) - 1, int(0.95 * len(values)))]


def summarize(dataset: str, tag: str):
    rows = []
    for prof, label in PROFILES:
        static_path = RESULTS / f"network/{prof}_{dataset}_{tag}_static.csv"
        if not static_path.exists():
            continue
        static = load(static_path)
        selector = load(RESULTS / f"network/{prof}_{dataset}_{tag}_selector.csv")
        ids = {r["question_id"] for r in static}
        for key, name in STRATEGIES:
            t = [float(r[f"{key}_s"]) for r in static]
            b = sum(float(r[f"{key}_bytes"]) for r in static)
            rows.append([label, name, len(t), f"{sum(t):.1f}", f"{p95(t):.2f}", f"{b / 1e6:.2f}"])
        for target in ("min_bytes", "min_latency"):
            sel = [r for r in selector if r["target"] == target and r["question_id"] in ids]
            t = [float(r["latency_s"]) for r in sel]
            b = sum(float(r["bytes"]) for r in sel)
            rows.append([label, f"selector {target}", len(t), f"{sum(t):.1f}", f"{p95(t):.2f}", f"{b / 1e6:.2f}"])
    return rows


def table(rows):
    header = ["profile", "strategy", "queries", "total latency (s)", "p95 (s)", "wire MB"]
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def speedups(dataset: str, tag: str):
    out = []
    for prof, label in PROFILES:
        static_path = RESULTS / f"network/{prof}_{dataset}_{tag}_static.csv"
        if not static_path.exists():
            continue
        static = load(static_path)
        selector = load(RESULTS / f"network/{prof}_{dataset}_{tag}_selector.csv")
        ids = {r["question_id"] for r in static}
        json_total = sum(float(r["json_s"]) for r in static)
        sel_total = sum(float(r["latency_s"]) for r in selector
                        if r["target"] == "min_latency" and r["question_id"] in ids)
        out.append(f"- {label}: selector (min_latency) {json_total / sel_total:.2f}x vs always JSON "
                   f"({json_total:.1f} s -> {sel_total:.1f} s)")
    return "\n".join(out)


def main():
    for dataset, tag, title in [("bird", "stride", "BIRD dev, every 5th query"),
                                ("bird", "large", "BIRD dev, the 5 largest results"),
                                ("spider", "stride", "Spider dev, every 5th query")]:
        rows = summarize(dataset, tag)
        if rows:
            print(f"### {title}\n")
            print(table(rows))
            print("\nSelector speed-up over always-JSON (total latency):\n")
            print(speedups(dataset, tag))
            print()


if __name__ == "__main__":
    main()
