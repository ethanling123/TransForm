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


TPCDS_LINKS = [("localhost", None), ("LAN 1 Gbit", 1000.0), ("WAN 50 Mbit", 50.0), ("cellular 10 Mbit", 10.0)]


def tpcds_section():
    """TPC-DS measured on localhost; slower links add bytes / bandwidth (modelled, no round trips) to each latency."""
    paths = {c: D / f"tpcds_{c}.csv" for c, _ in CONFIGS}
    if not all(p.exists() for p in paths.values()):
        return ""
    loaded = {c: {(r["n_rows"], r["n_cols"], r["strategy"]): r for r in load(p)} for c, p in paths.items()}
    keys = sorted(set.intersection(*(set(v) for v in loaded.values())))
    rows = []
    for strategy in sorted({k[2] for k in keys}):
        if strategy == "fixed:json":
            continue
        sel = [k for k in keys if k[2] == strategy]
        cells = []
        for link, mbps in TPCDS_LINKS:
            tot = []
            for c, _ in CONFIGS:
                tot.append(sum(float(loaded[c][k]["latency_s"]) + (int(loaded[c][k]["bytes"]) * 8 / (mbps * 1e6) if mbps else 0) for k in sel))
            cells.append(f"{tot[0]:.1f} / {tot[1]:.1f} ({tot[1] / tot[0]:.2f}x)")
        mb = [sum(int(loaded[c][k]["bytes"]) for k in sel) / 1e6 for c, _ in CONFIGS]
        rows.append([strategy, len(sel), f"{mb[0]:.0f} / {mb[1]:.0f}", *cells])
    header = ["strategy", "grid cells", "MB (snappy/none / zstd/zstd)", *[f"{n}: total s (snappy/none / zstd/zstd)" for n, _ in TPCDS_LINKS]]
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n## TPC-DS catalog_sales (localhost measured, slower links modelled)\n\n" + "\n".join(out) + "\n"


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
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    text = "# Table codec defaults: snappy/none vs zstd/zstd\n\n" + "\n".join(out) + "\n" + tpcds_section()
    (D / "defaults.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
