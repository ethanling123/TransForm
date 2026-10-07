"""BIRD dev through the server-side format selector (bird_query_run_inline)."""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from client.mcp_client import call_bird_query_run_inline, connect, fetch_blob, fetch_stream_chunks


def _via(url: str, port: int) -> str:
    return url.replace("localhost:8000", f"localhost:{port}").replace("127.0.0.1:8000", f"localhost:{port}")


def _read_parquet(b: bytes) -> int:
    return pq.read_table(io.BytesIO(b)).num_rows


def _read_ipc(b: bytes) -> int:
    return ipc.open_file(pa.BufferReader(b)).read_all().num_rows


async def run_query(session, client, port: int, q, target: str):
    t0 = time.perf_counter()
    res = await call_bird_query_run_inline(
        session, db_id=q["db_id"], sql=q["SQL"], optimization_target=target
    )
    fmt = res["chosen_format"]
    payload, decode = res["payload"], res["decode"]
    reader = _read_ipc if decode["encoding"] == "arrow_ipc" else _read_parquet

    if decode["transport"] == "inline":
        records = payload["records"]
        nbytes = len(json.dumps(records, separators=(",", ":")).encode())
        n_rows = len(records)
    elif decode["transport"] == "http_blob":
        data = await fetch_blob(client, _via(decode["url"], port))
        nbytes, n_rows = len(data), reader(data)
    else:
        nbytes = n_rows = 0
        async for chunk in fetch_stream_chunks(client, _via(decode["url"], port)):
            nbytes += len(chunk) + 8
            n_rows += reader(chunk)
    return {
        "question_id": q["question_id"],
        "target": target,
        "chosen_format": fmt,
        "bytes": nbytes,
        "latency_s": time.perf_counter() - t0,
        "n_rows": n_rows,
    }


def load_queries(path: str, sql_field: str):
    queries = json.loads(Path(path).read_text())
    for i, q in enumerate(queries):
        q.setdefault("question_id", str(i))
        q["SQL"] = q[sql_field]
    return queries


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="data/dev_20240627/dev.json")
    parser.add_argument("--targets", default="min_bytes,min_latency")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--ids", default=None)
    parser.add_argument("--exclude", default="")
    parser.add_argument("--sql-field", default="SQL")
    parser.add_argument("--out", default="results/bird/selector.csv")
    args = parser.parse_args()

    queries = load_queries(args.dev, args.sql_field)[: args.limit : args.stride]
    if args.ids:
        wanted = set(args.ids.split(","))
        queries = [q for q in load_queries(args.dev, args.sql_field) if str(q["question_id"]) in wanted]
    excluded = set(args.exclude.split(","))
    queries = [q for q in queries if str(q["question_id"]) not in excluded]
    rows, skipped = [], 0
    async with connect(f"http://localhost:{args.port}/mcp/mcp") as (session, client):
        for target in args.targets.split(","):
            for i, q in enumerate(queries):
                try:
                    rows.append(await run_query(session, client, args.port, q, target))
                except Exception as exc:
                    skipped += 1
                    print(f"skip {target} {q['question_id']}: {type(exc).__name__}", flush=True)
                if i % 200 == 0:
                    print(f"{target} {i}/{len(queries)}", flush=True)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"done: {len(rows)} ok, {skipped} skipped")


if __name__ == "__main__":
    asyncio.run(main())
