"""Synthetic format sweep: bytes, fetch+decode latency per wire format and size."""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import statistics
import time

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from client.mcp_client import (
    call_large_arrow_ipc_blob,
    call_large_arrow_ipc_stream,
    call_large_json,
    call_large_parquet_blob,
    call_large_parquet_stream,
    connect,
    fetch_blob,
    fetch_stream_chunks,
)

SIZES = [(100, 10), (1_000, 10), (10_000, 10), (100_000, 10), (500_000, 10)]
CHUNK_ROWS = 8192


def _json_bytes(records) -> int:
    import json

    return len(json.dumps(records, separators=(",", ":")).encode())


async def run_once(session, client, fmt: str, n_rows: int, n_cols: int):
    t0 = time.perf_counter()
    if fmt == "json":
        records = await call_large_json(session, n_rows, n_cols)
        nbytes = _json_bytes(records)
        decoded = len(records)
    elif fmt == "parquet_blob":
        desc = await call_large_parquet_blob(session, n_rows, n_cols)
        data = await fetch_blob(client, desc["url"])
        nbytes = len(data)
        decoded = pq.read_table(io.BytesIO(data)).num_rows
    elif fmt == "arrow_ipc_blob":
        desc = await call_large_arrow_ipc_blob(session, n_rows, n_cols)
        data = await fetch_blob(client, desc["url"])
        nbytes = len(data)
        decoded = ipc.open_file(pa.BufferReader(data)).read_all().num_rows
    else:
        if fmt == "parquet_stream":
            desc = await call_large_parquet_stream(session, n_rows, n_cols, CHUNK_ROWS)
            read = lambda b: pq.read_table(io.BytesIO(b)).num_rows
        else:
            desc = await call_large_arrow_ipc_stream(session, n_rows, n_cols, CHUNK_ROWS)
            read = lambda b: ipc.open_file(pa.BufferReader(b)).read_all().num_rows
        nbytes = 0
        decoded = 0
        async for chunk in fetch_stream_chunks(client, desc["url"]):
            nbytes += len(chunk) + 8
            decoded += read(chunk)
    elapsed = time.perf_counter() - t0
    if decoded != n_rows:
        raise RuntimeError(f"{fmt}: decoded {decoded} rows, expected {n_rows}")
    return nbytes, elapsed


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--out", default="results/synthetic/python.csv")
    args = parser.parse_args()

    formats = ["json", "parquet_blob", "parquet_stream", "arrow_ipc_blob", "arrow_ipc_stream"]
    rows = []
    async with connect() as (session, client):
        for n_rows, n_cols in SIZES:
            for fmt in formats:
                await run_once(session, client, fmt, n_rows, n_cols)  # warmup
                times, sizes = [], []
                for _ in range(args.reps):
                    nbytes, elapsed = await run_once(session, client, fmt, n_rows, n_cols)
                    times.append(elapsed)
                    sizes.append(nbytes)
                row = {
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "format": fmt,
                    "bytes": sizes[0],
                    "latency_median_s": round(statistics.median(times), 4),
                    "latency_min_s": round(min(times), 4),
                }
                rows.append(row)
                print(row, flush=True)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    asyncio.run(main())
