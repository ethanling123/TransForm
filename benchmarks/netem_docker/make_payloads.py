"""Static payloads for the protocol experiment: TPC-DS rows as zstd Parquet at about 100 KB, 1 MB and 8 MB, and the 1 MB
Parquet's rows as compact JSON (what the JSON tool would send for them).

    .venv/bin/python -m benchmarks.netem_docker.make_payloads data/tpcds_catalog_sales_sf1.parquet results/scratch/protocol_payloads
"""

from __future__ import annotations

import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

SIZES = {"p100k": 100_000, "p1m": 1_000_000, "p8m": 8_000_000}


def main(source: str, out: str) -> None:
    os.makedirs(out, exist_ok=True)
    pool = pa.Table.from_batches([next(pq.ParquetFile(source).iter_batches(batch_size=400_000))])
    sample = pa.BufferOutputStream()
    pq.write_table(pool.slice(0, 20_000), sample, compression="zstd")
    bytes_per_row = sample.getvalue().size / 20_000
    rows = {}
    for name, size in SIZES.items():
        n = min(pool.num_rows, max(1, round(size / bytes_per_row)))
        pq.write_table(pool.slice(0, n), os.path.join(out, f"{name}.parquet"), compression="zstd")
        rows[name] = n
    with open(os.path.join(out, "j1m.json"), "w") as f:
        json.dump(pool.slice(0, rows["p1m"]).to_pylist(), f, separators=(",", ":"), default=float)
    for fname in sorted(os.listdir(out)):
        print(f"{fname:16s} {os.path.getsize(os.path.join(out, fname)):>11,d} B")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
