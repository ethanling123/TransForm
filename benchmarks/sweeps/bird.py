"""BIRD dev sweep: per gold query, bytes and latency for JSON, Parquet and Arrow IPC."""

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

from client.mcp_client import (
    call_bird_query_json,
    call_bird_query_materialize,
    connect,
    fetch_blob,
)



async def measure_query(session, client, base_url: str, db_id: str, sql: str):
    row = {"db_id": db_id}

    t0 = time.perf_counter()
    records = await call_bird_query_json(session, db_id=db_id, sql=sql)
    if isinstance(records, dict) and "result" in records:
        records = records["result"]
    row["json_s"] = time.perf_counter() - t0
    row["json_bytes"] = len(json.dumps(records, separators=(",", ":")).encode())
    row["n_rows"] = len(records)

    t_mat = time.perf_counter()
    meta = await call_bird_query_materialize(session, db_id=db_id, sql=sql)
    row["materialize_s"] = time.perf_counter() - t_mat
    rid = meta["result_id"]
    row["n_cols"] = meta["n_cols"]

    t0 = time.perf_counter()
    data = await fetch_blob(client, f"{base_url}/blobs/{rid}.parquet")
    pq.read_table(io.BytesIO(data))
    row["parquet_fetch_s"] = time.perf_counter() - t0
    row["parquet_s"] = row["materialize_s"] + row["parquet_fetch_s"]
    row["parquet_bytes"] = len(data)

    t0 = time.perf_counter()
    data = await fetch_blob(client, f"{base_url}/ipc-blobs/{rid}.arrow")
    ipc.open_file(pa.BufferReader(data)).read_all()
    row["arrow_ipc_fetch_s"] = time.perf_counter() - t0
    row["arrow_ipc_s"] = row["materialize_s"] + row["arrow_ipc_fetch_s"]
    row["arrow_ipc_bytes"] = len(data)
    return row


def load_queries(path: str, sql_field: str):
    queries = json.loads(Path(path).read_text())
    for i, q in enumerate(queries):
        q.setdefault("question_id", str(i))
        q["SQL"] = q[sql_field]
    return queries


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="data/dev_20240627/dev.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--ids", default=None)
    parser.add_argument("--exclude", default="")
    parser.add_argument("--sql-field", default="SQL")
    parser.add_argument("--out", default="results/bird/static.csv")
    args = parser.parse_args()

    queries = load_queries(args.dev, args.sql_field)[: args.limit : args.stride]
    if args.ids:
        wanted = set(args.ids.split(","))
        queries = [q for q in load_queries(args.dev, args.sql_field) if str(q["question_id"]) in wanted]
    excluded = set(args.exclude.split(","))
    queries = [q for q in queries if str(q["question_id"]) not in excluded]
    rows, skipped = [], 0
    base_url = f"http://localhost:{args.port}"
    async with connect(f"{base_url}/mcp/mcp") as (session, client):
        for i, q in enumerate(queries):
            try:
                row = await measure_query(session, client, base_url, q["db_id"], q["SQL"])
            except Exception as exc:
                skipped += 1
                print(f"skip {q['question_id']}: {type(exc).__name__}: {str(exc)[:80]}", flush=True)
                continue
            row["question_id"] = q["question_id"]
            rows.append(row)
            if i % 100 == 0:
                print(f"{i}/{len(queries)}", flush=True)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"done: {len(rows)} ok, {skipped} skipped")


if __name__ == "__main__":
    asyncio.run(main())
