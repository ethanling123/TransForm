"""TPC-DS catalog_sales before and after the stream arms stopped converting a pandas slice per chunk.

Before: results/defaults/tpcds_zstd_zstd.csv (same server settings, previous commit). After: results/data_plane/tpcds_streams_after.csv
(benchmarks/scripts/run_tpcds_once.sh). Sums are over the nine grid cells (10K, 100K, 500K rows x 6, 20, 34 columns); the
largest cell is shown on its own.

    .venv/bin/python -m benchmarks.analysis.stream_fix
"""

from __future__ import annotations

import csv


def load(path):
    return {(r["n_rows"], r["n_cols"], r["strategy"]): r for r in csv.DictReader(open(path))}


def main():
    before, after = load("results/defaults/tpcds_zstd_zstd.csv"), load("results/data_plane/tpcds_streams_after.csv")
    keys = sorted(set(before) & set(after))
    rows = []
    for strategy in ("fixed:parquet_stream", "fixed:arrow_ipc_stream", "selector:min_time_to_first_rows", "fixed:parquet_blob", "fixed:arrow_ipc_blob"):
        ks = [k for k in keys if k[2] == strategy]
        big = next(k for k in ks if k[0] == "500000" and k[1] == "34")
        f = lambda d, col, sel=ks: sum(float(d[k][col] or 0) for k in sel)
        rows.append([strategy.replace("fixed:", ""), f"{f(before, 'latency_s'):.2f} s", f"{f(after, 'latency_s'):.2f} s",
                     f"{f(before, 'ttfr_s'):.2f} s", f"{f(after, 'ttfr_s'):.2f} s",
                     f"{float(before[big]['latency_s']):.2f} s", f"{float(after[big]['latency_s']):.2f} s",
                     f"{sum(int(before[k]['bytes']) for k in ks) / 1e6:.1f} MB", f"{sum(int(after[k]['bytes']) for k in ks) / 1e6:.1f} MB"])
    header = ["arm", "total before", "total after", "time to first rows before", "after", "500K x 34 before", "after", "bytes before", "after"]
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]
    print("\n".join(out))


if __name__ == "__main__":
    main()
