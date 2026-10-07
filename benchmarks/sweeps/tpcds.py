"""TPC-DS catalog_sales through the MCP server: fixed formats and the selector.

Grid follows the original harness: first N rows x first K columns, N in {10K,100K,500K}, K in {6,20,34}.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import time

import httpx
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from client.mcp_client import (
    call_large_arrow_ipc_blob,
    call_large_arrow_ipc_stream,
    call_large_json,
    call_large_parquet_blob,
    call_large_parquet_stream,
    call_large_result_auto,
    connect,
    fetch_blob,
    fetch_stream_chunks,
    register_materialized,
)

ROWS = [10_000, 100_000, 500_000]
COLS = [6, 20, 34]
CHUNK_ROWS = 8192
JSON_CELL_CAP = 5_000_000  # server's MAX_JSON_CELLS default


def _via(url: str, port: int) -> str:
    return url.replace("localhost:8000", f"localhost:{port}").replace("127.0.0.1:8000", f"localhost:{port}")


async def fetch_descriptor(client, port, desc, fmt):
    """Fetch and decode; return (bytes, rows, first_chunk_bytes, ttfr_s)."""
    t0 = time.perf_counter()
    reader = (lambda b: ipc.open_file(pa.BufferReader(b)).read_all().num_rows) if "arrow" in fmt else (
        lambda b: pq.read_table(io.BytesIO(b)).num_rows
    )
    if fmt.endswith("stream"):
        total = rows = 0
        first = ttfr = None
        async for chunk in fetch_stream_chunks(client, _via(desc["url"], port)):
            if first is None:
                first, ttfr = len(chunk) + 8, time.perf_counter() - t0
            total += len(chunk) + 8
            rows += reader(chunk)
        return total, rows, first, ttfr
    data = await fetch_blob(client, _via(desc["url"], port))
    rows = reader(data)
    return len(data), rows, None, time.perf_counter() - t0


async def run_format(session, client, port, fmt, rid, n_rows, n_cols):
    t0 = time.perf_counter()
    if fmt == "json":
        records = await call_large_json(session, n_rows, n_cols, result_id=rid)
        return {"bytes": len(json.dumps(records, separators=(",", ":")).encode()),
                "rows": len(records), "first_chunk_bytes": "", "ttfr_s": time.perf_counter() - t0,
                "latency_s": time.perf_counter() - t0}
    calls = {
        "parquet_blob": lambda: call_large_parquet_blob(session, n_rows, n_cols, result_id=rid),
        "arrow_ipc_blob": lambda: call_large_arrow_ipc_blob(session, n_rows, n_cols, result_id=rid),
        "parquet_stream": lambda: call_large_parquet_stream(session, n_rows, n_cols, CHUNK_ROWS, result_id=rid),
        "arrow_ipc_stream": lambda: call_large_arrow_ipc_stream(session, n_rows, n_cols, CHUNK_ROWS, result_id=rid),
    }
    desc = await calls[fmt]()
    nbytes, rows, first, ttfr = await fetch_descriptor(client, port, desc, fmt)
    return {"bytes": nbytes, "rows": rows, "first_chunk_bytes": first if first is not None else "",
            "ttfr_s": ttfr, "latency_s": time.perf_counter() - t0}


async def run_selector(session, client, port, target, rid, n_rows, n_cols):
    t0 = time.perf_counter()
    res = await call_large_result_auto(
        session, n_rows, n_cols, CHUNK_ROWS, result_id=rid, optimization_target=target,
        prefer_streaming=(target == "min_time_to_first_rows"),
    )
    fmt, decode = res["chosen_format"], res["decode"]
    if decode["transport"] == "inline":
        records = res["payload"]["records"]
        return fmt, {"bytes": len(json.dumps(records, separators=(",", ":")).encode()), "rows": len(records),
                     "first_chunk_bytes": "", "ttfr_s": time.perf_counter() - t0,
                     "latency_s": time.perf_counter() - t0}
    nbytes, rows, first, ttfr = await fetch_descriptor(client, port, decode, fmt)
    return fmt, {"bytes": nbytes, "rows": rows, "first_chunk_bytes": first if first is not None else "",
                 "ttfr_s": ttfr, "latency_s": time.perf_counter() - t0}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default="data/tpcds_catalog_sales_sf1.parquet")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--out", default="results/tpcds/python.csv")
    args = parser.parse_args()

    base = pq.read_table(args.parquet)
    fixed = ["json", "parquet_blob", "parquet_stream", "arrow_ipc_blob", "arrow_ipc_stream"]
    targets = ["min_bytes", "min_latency", "min_time_to_first_rows"]
    rows_out = []
    base_url = f"http://localhost:{args.port}"
    async with connect(f"{base_url}/mcp/mcp") as (session, client):
        client.timeout = httpx.Timeout(600.0)
        for n_rows in ROWS:
            for n_cols in COLS:
                sub = base.slice(0, n_rows).select(base.column_names[:n_cols])
                buf = io.BytesIO()
                pq.write_table(sub, buf)
                rid = (await register_materialized(client, buf.getvalue(), base_url=base_url))["result_id"]
                runs = [(f, f"fixed:{f}") for f in fixed] + [(t, f"selector:{t}") for t in targets]
                for key, label in runs:
                    results = []
                    if key == "json" and n_rows * n_cols > JSON_CELL_CAP:
                        row = {"n_rows": n_rows, "n_cols": n_cols, "strategy": label, "chosen": "capped (>5M cells)",
                               "bytes": "", "first_chunk_bytes": "", "latency_s": "", "ttfr_s": ""}
                        rows_out.append(row)
                        print(row, flush=True)
                        continue
                    try:
                        for _ in range(args.reps + 1):  # first run is warmup
                            if label.startswith("fixed"):
                                r = await run_format(session, client, args.port, key, rid, n_rows, n_cols)
                                chosen = key
                            else:
                                chosen, r = await run_selector(session, client, args.port, key, rid, n_rows, n_cols)
                            assert r["rows"] == n_rows, (label, r["rows"], n_rows)
                            results.append(r)
                    except Exception as exc:  # e.g. inline JSON above the server's 5M-cell cap
                        row = {"n_rows": n_rows, "n_cols": n_cols, "strategy": label, "chosen": f"error: {type(exc).__name__}",
                               "bytes": "", "first_chunk_bytes": "", "latency_s": "", "ttfr_s": ""}
                        rows_out.append(row)
                        print(row, flush=True)
                        continue
                    results = results[1:]
                    results.sort(key=lambda r: r["latency_s"])
                    med = results[len(results) // 2]
                    row = {"n_rows": n_rows, "n_cols": n_cols, "strategy": label, "chosen": chosen,
                           "bytes": med["bytes"], "first_chunk_bytes": med["first_chunk_bytes"],
                           "latency_s": round(med["latency_s"], 4), "ttfr_s": round(med["ttfr_s"], 4)}
                    rows_out.append(row)
                    print(row, flush=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)


if __name__ == "__main__":
    asyncio.run(main())
