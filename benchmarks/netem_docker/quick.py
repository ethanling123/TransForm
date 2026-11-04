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


async def raw_url(session, rid):
    """The signed raw-blob URL for a registered payload, from a descriptor rather than built by hand."""
    out = await session.call_tool("fetch_arm", arguments={"arm": "raw_blob", "result_id": rid})
    return out.structuredContent["url"]


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
        if out.get("decode", {}).get("arm"):
            chosen += ":" + out["decode"]["arm"]
        if chosen == "json":
            return len(json.dumps(out["payload"], separators=(",", ":")).encode()), -1, chosen
        if "inline_bytes" in out:
            return len(out["inline_bytes"]), -1, chosen + " inline"
        url = out["decode"]["url"]
        if out["chosen_format"].endswith("_stream"):
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
    parser.add_argument("--breakdown", action="store_true", help="only time the pieces of one small blob fetch")
    parser.add_argument("--learn", action="store_true", help="report a few downloads to the server, then let the selector choose")
    parser.add_argument("--ranges", action="store_true", help="download one 8 MB payload over 1, 2, 4 and 8 connections")
    parser.add_argument("--conditional", action="store_true", help="synthetic: fetch a result, then fetch it again with If-None-Match")
    parser.add_argument("--tables", action="store_true", help="only time the selector on three larger uploaded tables")
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
        link_url = await raw_url(session, rid)
        t = time.perf_counter()
        resp = await client.get(link_url)
        download_s = time.perf_counter() - t
        record("link", "ping rtt (median)", "", 0, statistics.median(pings))
        record("link", f"4 MB download = {len(resp.content) * 8 / download_s / 1e6:.1f} Mbit/s", "", len(resp.content), download_s)

        if args.breakdown:
            await breakdown(session, client, record)
            return _write(args.out, rows)
        if args.learn:
            await learn_link(session, client, record)
            return _write(args.out, rows)
        if args.conditional:
            await conditional_experiment(session, client, record, args.reps)
            return _write(args.out, rows)
        if args.ranges:
            await ranges_experiment(session, client, record, args.reps)
            return _write(args.out, rows)
        if args.tables:
            await larger_tables(session, client, record, args.reps)
            return _write(args.out, rows)
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

    _write(args.out, rows)


async def conditional_experiment(session, client, record, reps):
    """Synthetic repeated fetch of one 4.5 MB table: unconditional repeat against a repeat that sends If-None-Match."""
    table = uploaded_table(200_000)
    up = await register_materialized(client, _parquet(table))
    desc = (await call_large_parquet_blob(session, up["n_rows"], up["n_cols"], result_id=up["result_id"]))["url"]
    first = await client.get(desc)
    etag = first.headers["etag"]
    plain, conditional = [], []
    for _ in range(reps):
        t = time.perf_counter()
        body = (await client.get(desc)).content
        plain.append(time.perf_counter() - t)
        t = time.perf_counter()
        resp = await client.get(desc, headers={"If-None-Match": etag})
        conditional.append(time.perf_counter() - t)
        assert resp.status_code == 304 and resp.content == b""
    record("conditional fetch", "repeat, unconditional", "", len(body), statistics.median(plain))
    record("conditional fetch", "repeat, If-None-Match (304)", "", 0, statistics.median(conditional))


async def ranges_experiment(session, client, record, reps):
    """One payload fetched as N byte ranges over N separate connections; N=0 is a plain GET on the shared connection."""
    data = os.urandom(8_000_000)
    rid = (await client.post("http://localhost:8000/materialized-raw", content=data,
                             headers={"Content-Type": "application/octet-stream"})).json()["result_id"]
    url = await raw_url(session, rid)

    async def part(start, end):
        async with httpx.AsyncClient(timeout=600) as c:
            return (await c.get(url, headers={"Range": f"bytes={start}-{end}"})).content

    for n in (0, 1, 2, 4, 8):
        times = []
        for _ in range(reps):
            t = time.perf_counter()
            if n == 0:
                body = (await client.get(url)).content
            else:
                step = -(-len(data) // n)
                body = b"".join(await asyncio.gather(*[part(a, min(a + step, len(data)) - 1) for a in range(0, len(data), step)]))
            times.append(time.perf_counter() - t)
            assert body == data
        label = "plain GET" if n == 0 else f"{n} range request{'s' if n > 1 else ''}"
        record("ranges 8 MB", f"{label} = {len(data) * 8 / statistics.median(times) / 1e6:.1f} Mbit/s", "", len(data), statistics.median(times))


async def learn_link(session, client, record):
    """Report downloads to record_format_outcome, then run the unstructured selector with whatever the server learned."""
    rid = (await client.post("http://localhost:8000/materialized-raw", content=os.urandom(1_000_000),
                             headers={"Content-Type": "application/octet-stream"})).json()["result_id"]
    estimate = None
    url = await raw_url(session, rid)
    for _ in range(4):
        t = time.perf_counter()
        body = (await client.get(url)).content
        seconds = time.perf_counter() - t
        out = await session.call_tool("record_format_outcome", arguments={
            "n_rows": 0, "n_cols": 0, "optimization_target": "min_latency", "format_used": "raw_blob",
            "bytes": len(body), "latency_s": seconds})
        estimate = (out.structuredContent or {}).get("link_mbps_estimate")
    record("learned link", f"server estimate = {estimate if estimate is None else round(estimate, 1)} Mbit/s", "", 0, 0.0)
    ns = Namespace(server_port=8000, port=8000)
    text = b"".join(b'{"id":%d,"user":"user-%d","event":"click","path":"/item/%d"}\n' % (i, i % 300, i % 977) for i in range(9_000))
    for name, data, mime in (("text 600 KB (compressible)", text, "text/plain"), ("bytes 1 MB (incompressible)", os.urandom(1_000_000), "application/octet-stream")):
        for _ in range(2):
            label, wire, latency, _, chosen, ok = await run_selector(session, client, ns, name, data, mime, "min_latency")
        record(name, "selector min_latency", chosen, wire, latency)


def larger_tables(session, client, record, reps):
    return _larger_tables(session, client, record, reps)


async def _larger_tables(session, client, record, reps):
    """Selector (min_latency) on tables big enough that the codec choice matters."""
    rng = np.random.default_rng(3)
    n = 200_000
    shapes = {
        f"mixed {n}x6": uploaded_table(n),
        "numeric 50000x20 random": pa.table({f"c{i}": (rng.random(50_000) if i % 2 else rng.integers(0, 10**9, 50_000)) for i in range(20)}),
        f"categorical {n}x8": pa.table({f"k{i}": np.array([f"value-{j}" for j in range(30)])[rng.integers(0, 30, n)] for i in range(8)}),
    }
    for label, table in shapes.items():
        up = await register_materialized(client, _parquet(table))
        fn = [x for x in await table_formats(session, client, up["result_id"], up["n_rows"], up["n_cols"]) if x[0].startswith("selector")][0][1]
        await timed(fn, up["n_rows"])
        runs = [await timed(fn, up["n_rows"]) for _ in range(reps)]
        record(f"tables {label}", "selector min_latency", runs[-1][2], runs[0][0], statistics.median(r[1] for r in runs))


def _write(out, rows):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["profile", "workload", "strategy", "arm", "wire_bytes", "seconds"])
        w.writerows(rows)


async def breakdown(session, client, record):
    """Where the time of a small result goes: the tool call, the first fetch, a repeat fetch on the same connection,
    and a fetch on a brand-new connection (the cost an inline arm would avoid)."""
    for label, n in (("small 2K rows", 2_000), ("5K rows", 5_000), ("10K rows", 10_000), ("medium 20K rows", 20_000)):
        up = await register_materialized(client, _parquet(uploaded_table(n)))
        rid = up["result_id"]
        await call_large_parquet_blob(session, n, 6, result_id=rid)  # warm the server's caches
        calls, firsts, repeats, fresh = [], [], [], []
        for _ in range(5):
            t = time.perf_counter()
            desc = await call_large_parquet_blob(session, n, 6, result_id=rid)
            calls.append(time.perf_counter() - t)
            t = time.perf_counter()
            data = await fetch_blob(client, desc["url"])
            firsts.append(time.perf_counter() - t)
            t = time.perf_counter()
            await fetch_blob(client, desc["url"])
            repeats.append(time.perf_counter() - t)
            async with httpx.AsyncClient(timeout=60) as new_client:
                t = time.perf_counter()
                await fetch_blob(new_client, desc["url"])
                fresh.append(time.perf_counter() - t)
        pings = []
        for _ in range(5):
            t = time.perf_counter()
            await session.send_ping()
            pings.append(time.perf_counter() - t)
        m = statistics.median
        record(f"breakdown {label}", "mcp ping", "", 0, m(pings))
        record(f"breakdown {label}", "tool call (descriptor)", "", 0, m(calls))
        record(f"breakdown {label}", "fetch, reused connection", "", len(data), m(firsts))
        record(f"breakdown {label}", "fetch, repeat same connection", "", len(data), m(repeats))
        record(f"breakdown {label}", "fetch, new connection", "", len(data), m(fresh))
        selector = [x for x in await table_formats(session, client, rid, n, 6) if x[0].startswith("selector")][0][1]
        runs = [await timed(selector, n) for _ in range(5)]
        record(f"breakdown {label}", "selector call + fetch/inline", runs[-1][2], runs[-1][0], m(r[1] for r in runs))


def _parquet(table: pa.Table) -> bytes:
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


if __name__ == "__main__":
    asyncio.run(main())
