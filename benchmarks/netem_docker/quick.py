"""A few-minute check of the formats across a real (kernel-shaped) network link between two machines.

Run by compose.yml inside the client container, with localhost:8000 forwarded to the server. It measures:

  link      round-trip time of an MCP ping and the throughput of one 4 MB download, to confirm the shaping took effect
  tables    one table the server generates and one uploaded table with strings, floats and timestamps, each as JSON,
            Parquet blob and Arrow IPC blob, plus the selector (min_latency)
  payloads  compressible text and incompressible bytes as raw, gzip, zstd and brotli blobs, plus the selector

Each row is the median of REPS timed runs after one warm-up; bytes are what crossed the connection (Parquet and Arrow
inline sizes) and every download is decoded and checked.

    python -m benchmarks.netem_docker.quick --out results/netem_docker/wan.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import os
import statistics
import time
from argparse import Namespace

import httpx
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from benchmarks.sweeps.payloads import run_forced, run_selector
from client.mcp_client import (
    call_large_arrow_ipc_blob,
    call_large_json,
    call_large_parquet_blob,
    call_large_result_auto,
    connect,
    fetch_blob,
    fetch_stream_chunks,
    register_materialized,
)

REPS = 2  # --reps overrides
SYNTHETIC_ROWS, UPLOADED_ROWS = 50_000, 50_000


def uploaded_table(n: int) -> pa.Table:
    rng = np.random.default_rng(0)
    words = np.array([f"category-{i}" for i in range(40)])
    names = ["".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), 12)) for _ in range(n)]
    return pa.table({
        "id": np.arange(n),
        "price": rng.random(n) * 1000,
        "quantity": rng.integers(0, 500, n),
        "category": words[rng.integers(0, 40, n)],
        "name": names,
        "created": (1_700_000_000 + rng.integers(0, 90_000_000, n)).astype("datetime64[s]"),
    })


async def table_formats(session, client, rid, n_rows, n_cols):
    """(label, fetch-and-decode coroutine) per format for one table; rid is None for a server-generated table."""
    kw = {} if rid is None else {"result_id": rid}

    async def json_fmt():
        records = await call_large_json(session, n_rows, n_cols, **kw)
        return len(json.dumps(records, separators=(",", ":")).encode()), len(records)

    async def parquet_fmt():
        data = await fetch_blob(client, (await call_large_parquet_blob(session, n_rows, n_cols, **kw))["url"])
        return len(data), pq.read_table(io.BytesIO(data)).num_rows

    async def arrow_fmt():
        data = await fetch_blob(client, (await call_large_arrow_ipc_blob(session, n_rows, n_cols, **kw))["url"])
        return len(data), ipc.open_file(pa.BufferReader(data)).read_all().num_rows

    async def selector():
        out = await call_large_result_auto(session, n_rows, n_cols, result_id=rid, optimization_target="min_latency")
        chosen = out["chosen_format"]
        if chosen == "json":
            return len(json.dumps(out["payload"], separators=(",", ":")).encode()), -1, chosen
        url = out["decode"]["url"]
        if chosen.endswith("_stream"):
            total = sum([len(c) + 8 async for c in fetch_stream_chunks(client, url)])
            return total, -1, chosen
        return len(await fetch_blob(client, url)), -1, chosen

    return [("json", json_fmt), ("parquet_blob", parquet_fmt), ("arrow_ipc_blob", arrow_fmt), ("selector min_latency", selector)]


async def timed(fn, expect_rows):
    t = time.perf_counter()
    result = await fn()
    elapsed = time.perf_counter() - t
    nbytes, rows, *chosen = result
    if rows >= 0 and rows != expect_rows:
        raise RuntimeError(f"decoded {rows} rows, expected {expect_rows}")
    return nbytes, elapsed, (chosen[0] if chosen else "")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/netem_docker/quick.csv")
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--no-json", action="store_true", help="skip JSON, which is slow on a slow link")
    args = parser.parse_args()
    profile = os.environ.get("PROFILE", "unshaped")
    rows = []

    def record(workload, strategy, arm, nbytes, seconds):
        rows.append([profile, workload, strategy, arm, nbytes, round(seconds, 4)])
        print(f"{profile:10s} {workload:28s} {strategy:22s} {arm:14s} {nbytes:>11,d} B {seconds:8.3f} s", flush=True)

    async with connect("http://localhost:8000/mcp/mcp") as (session, client):
        client.timeout = httpx.Timeout(600.0)

        pings = []
        for _ in range(10):
            t = time.perf_counter()
            await session.send_ping()
            pings.append(time.perf_counter() - t)
        rid = (await client.post("http://localhost:8000/materialized-raw", content=os.urandom(4_000_000),
                                 headers={"Content-Type": "application/octet-stream"})).json()["result_id"]
        t = time.perf_counter()
        resp = await client.get(f"http://localhost:8000/raw/{rid}")
        download_s = time.perf_counter() - t
        record("link", "ping rtt (median)", "", 0, statistics.median(pings))
        record("link", f"4 MB download = {len(resp.content) * 8 / download_s / 1e6:.1f} Mbit/s", "", len(resp.content), download_s)

        up = await register_materialized(client, _parquet(uploaded_table(UPLOADED_ROWS)))
        for workload, rid, n_rows, n_cols in ((f"generated {SYNTHETIC_ROWS}x10", None, SYNTHETIC_ROWS, 10),
                                              (f"uploaded {UPLOADED_ROWS}x6 mixed", up["result_id"], up["n_rows"], up["n_cols"])):
            for label, fn in await table_formats(session, client, rid, n_rows, n_cols):
                if args.no_json and label == "json":
                    continue
                await timed(fn, n_rows)
                runs = [await timed(fn, n_rows) for _ in range(args.reps)]
                record(workload, label, runs[-1][2], runs[0][0], statistics.median(r[1] for r in runs))

        text = b"".join(b'{"id":%d,"user":"user-%d","event":"click","path":"/item/%d"}\n' % (i, i % 300, i % 977)
                        for i in range(9_000))
        noise = os.urandom(1_000_000)
        ns = Namespace(server_port=8000, port=8000)
        for name, data, mime in (("text 600 KB (compressible)", text, "text/plain"),
                                 ("bytes 1 MB (incompressible)", noise, "application/octet-stream")):
            for strategy in ("raw", "gzip", "zstd", "brotli"):
                runs = []
                for _ in range(1 + args.reps):
                    arm, wire, latency, _, ok = await run_forced(session, client, ns, name, data, mime, strategy)
                    assert ok
                    runs.append((wire, latency))
                record(name, strategy, "", runs[-1][0], statistics.median(r[1] for r in runs[1:]))
            runs = []
            for _ in range(1 + args.reps):
                label, wire, latency, _, chosen, ok = await run_selector(session, client, ns, name, data, mime, "min_latency")
                assert ok
                runs.append((wire, latency, chosen))
            record(name, "selector min_latency", runs[-1][2], runs[-1][0], statistics.median(r[1] for r in runs[1:]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["profile", "workload", "strategy", "arm", "wire_bytes", "seconds"])
        w.writerows(rows)


def _parquet(table: pa.Table) -> bytes:
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


if __name__ == "__main__":
    asyncio.run(main())
