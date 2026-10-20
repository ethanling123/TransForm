"""Measured comparison of the cost-aware selector (results/payloads_cost_aware/) against the legacy selector
(results/payloads/) and the forced arms, per link and payload kind.

Both sweeps force every arm on every payload in the same run, so each cost-aware CSV carries its own baseline. Totals are
summed latency; `regret` is that total divided by the per-payload best forced arm of the same run. The localhost run is
told the link is 20 Gbit/s; the legacy localhost run had no link speed set.

    .venv/bin/python -m benchmarks.analysis.cost_aware_e2e
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

R = Path("results")
LINKS = ["localhost", "lan", "wan", "cellular"]
ARMS = ["raw", "gzip", "zstd", "brotli"]


def load(path):
    out = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["payload"]][r["strategy"]] = r
    return out


def main():
    lines = ["# Cost-aware selector, measured end to end", "",
             "Regret = total latency / total of the best forced arm per payload, same run. `legacy min_latency` is the "
             "selector without the flag (results/payloads/); `cost-aware min_latency` is the flagged run. "
             "`select_s` is the mean time the tool call spent choosing.", "",
             "| link | kind (n) | legacy min_latency | cost-aware min_latency | best fixed arm | cost-aware picks | "
             "legacy select_s | cost-aware select_s | round trips ok |", "|---|---|---|---|---|---|---|---|---|"]
    for link in LINKS:
        new_path, old_path = R / "payloads_cost_aware" / f"{link}.csv", R / "payloads" / f"{link}.csv"
        if not (new_path.exists() and old_path.exists()):
            continue
        new, old = load(new_path), load(old_path)
        for kind in ("image", "audio", "text"):
            names = [p for p, v in new.items() if v["raw"]["kind"] == kind and p in old]
            if not names:
                continue
            oracle = sum(min(float(new[p][a]["latency_s"]) for a in ARMS) for p in names)
            fixed = {a: sum(float(new[p][a]["latency_s"]) for p in names) for a in ARMS}
            best = min(fixed, key=fixed.get)
            legacy = sum(float(old[p]["selector min_latency"]["latency_s"]) for p in names)
            ours = sum(float(new[p]["selector min_latency"]["latency_s"]) for p in names)
            picks = defaultdict(int)
            for p in names:
                picks[new[p]["selector min_latency"]["arm"]] += 1
            sel_old = sum(float(old[p]["selector min_latency"]["select_s"]) for p in names) / len(names)
            sel_new = sum(float(new[p]["selector min_latency"]["select_s"]) for p in names) / len(names)
            ok = all(r["ok"] == "True" for p in names for r in new[p].values())
            lines.append(f"| {link} | {kind} ({len(names)}) | {legacy / oracle:.2f}x | {ours / oracle:.2f}x | "
                         f"{fixed[best] / oracle:.2f}x ({best}) | " + ", ".join(f"{k}: {v}" for k, v in sorted(picks.items()))
                         + f" | {sel_old * 1000:.0f} ms | {sel_new * 1000:.0f} ms | {'yes' if ok else 'NO'} |")
    text = "\n".join(lines) + "\n"
    Path("results/payloads_cost_aware/summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
