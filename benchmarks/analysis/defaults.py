"""Compare the server's table codec defaults (results/defaults/): snappy Parquet + uncompressed Arrow IPC against zstd + zstd.

Same BIRD queries, same server, same link; only PARQUET_COMPRESSION and ARROW_IPC_COMPRESSION differ. For each link
and query set: total latency, p95 and wire bytes of each fixed format, and of the selector (which sizes its hints
with the server's default codecs, so the default also changes what it picks).

    .venv/bin/python -m benchmarks.analysis.defaults
"""

from __future__ import annotations

import csv
from pathlib import Path

D = Path("results/defaults")
CONFIGS = [("snappy_none", "snappy / none"), ("zstd_zstd", "zstd / zstd")]
LINKS = [("localhost", "localhost", "all"), ("lan", "LAN (1 ms, 1 Gbit)", "stride"), ("lan", "LAN, results over 50K cells", "large"),
         ("wan", "WAN (40 ms, 50 Mbit)", "stride"), ("wan", "WAN, results over 50K cells", "large"),
         ("cellular", "Cellular (80 ms, 10 Mbit)", "stride"), ("cellular", "Cellular, results over 50K cells", "large")]
FIXED = [("json", "JSON"), ("parquet", "Parquet"), ("arrow_ipc", "Arrow IPC")]


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def p95(values):
    values = sorted(values)
    return values[min(len(values) - 1, int(0.95 * len(values)))]


def main():
    rows = []
    for link, label, tag in LINKS:
        paths = {c: (D / f"{c}_{link}_{tag}_static.csv", D / f"{c}_{link}_{tag}_selector.csv") for c, _ in CONFIGS}
        if not all(p.exists() for pair in paths.values() for p in pair):
            continue
        loaded = {c: (load(s), load(sel)) for c, (s, sel) in paths.items()}
        ids = set.intersection(*({r["question_id"] for r in static} for static, _ in loaded.values()))
        for key, name in FIXED[1:] + [("selector:min_bytes", "selector min_bytes"), ("selector:min_latency", "selector min_latency")]:
            cells = []
            for c, _ in CONFIGS:
                static, selector = loaded[c]
                if key.startswith("selector"):
                    target = key.split(":")[1]
                    sel = [r for r in selector if r["target"] == target and r["question_id"] in ids]
                    t, b = [float(r["latency_s"]) for r in sel], sum(float(r["bytes"]) for r in sel)
                else:
                    sel = [r for r in static if r["question_id"] in ids]
                    t, b = [float(r[f"{key}_s"]) for r in sel], sum(float(r[f"{key}_bytes"]) for r in sel)
                cells.append((sum(t), p95(t), b / 1e6))
            (t0, p0, b0), (t1, p1, b1) = cells
            rows.append([label, len(ids), name, f"{t0:.1f}", f"{t1:.1f}", f"{t1 / t0:.2f}x", f"{p0:.2f}", f"{p1:.2f}",
                         f"{b0:.1f}", f"{b1:.1f}", f"{b1 / b0:.2f}x"])
    header = ["link", "queries", "strategy", "total s (snappy/none)", "total s (zstd/zstd)", "ratio", "p95 s (snappy/none)",
              "p95 s (zstd/zstd)", "MB (snappy/none)", "MB (zstd/zstd)", "ratio"]
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]
    text = "# Table codec defaults: snappy/none vs zstd/zstd\n\n" + "\n".join(out) + "\n"
    (D / "defaults.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
