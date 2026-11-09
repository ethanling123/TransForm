"""Time to first rows, early termination and an Arrow Flight baseline over a real (kernel-shaped) link.

Run by compose_transport.yml inside the client container; the server container runs the probe server (TransForm, with a
chunk counter) and the Flight server. Every arm fetches the same table in 8192-row units and stops once it has decoded
a target share of the rows (the first unit, 10%, 50%, or all):

  parquet_stream, arrow_stream   TransForm HTTP chunk streams (length-prefixed micro-Parquet / micro-IPC files)
  parquet_blob, arrow_blob       whole-file downloads; the first rows exist only after the whole download and decode
  flight_{none,lz4,zstd}         Arrow Flight DoGet with the ticket already in hand (as it would be in an MCP descriptor)
  flight_zstd_info               the same with a GetFlightInfo round trip first
  json                           the large_json tool (--json; the table is far too large for slow links)

Every fetch uses a fresh connection (cold); the MCP session that mints descriptors is warm and is timed separately
(call_s). Bytes are read from the client's NIC receive counter, so they include headers, TLS-free framing and anything the
server kept sending after the client stopped: rx_cut when the target was reached, rx_settled once the counter has been
still for 300 ms. server_units is how many chunks or batches the server produced for the fetch.

    python -m benchmarks.netem_docker.transport --out results/netem_docker/transport_wan.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import math
import os
import random
import statistics
import time

import httpx
import pyarrow as pa
import pyarrow.flight as flight
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from benchmarks.netem_docker.transport_common import FLIGHT_PORT, ROWS_PER_CHUNK, tables
from client.mcp_client import (
    call_large_arrow_ipc_blob,
    call_large_arrow_ipc_stream,
    call_large_json,
    call_large_parquet_blob,
    call_large_parquet_stream,
    connect,
    register_materialized,
)

SERVER_HOST = os.environ.get("SERVER_HOST", "localhost")
BASE = f"http://{SERVER_HOST}:8000"
CUTS = {"first": None, "10%": 0.1, "50%": 0.5, "100%": 1.0}
FLIGHT_CODECS = ("none", "lz4", "zstd")
RX_PATH = "/sys/class/net/eth0/statistics/rx_bytes"
TRACE = bool(os.environ.get("TRANSPORT_TRACE"))


def nic_rx() -> int:
    try:
        with open(RX_PATH) as f:
            return int(f.read())
    except OSError:
        return 0


async def settle(quiet_s: float = 0.3, limit_s: float = 20.0) -> None:
    started = time.perf_counter()
    deadline = started + limit_s
    last, since = nic_rx(), started
    while time.perf_counter() < deadline:
        await asyncio.sleep(0.05)
        now = nic_rx()
        if now != last:
            last, since = now, time.perf_counter()
        elif time.perf_counter() - since >= quiet_s:
            break
    if TRACE:
        print(f"    settle {time.perf_counter() - started:5.1f} s", flush=True)


def target_rows(cut: str, n_rows: int) -> int:
    frac = CUTS[cut]
    return 1 if frac is None else math.ceil(frac * n_rows)


def _decode_parquet_chunk(chunk: bytes) -> int:
    return pq.read_table(io.BytesIO(chunk)).num_rows


def _decode_ipc_chunk(chunk: bytes) -> int:
    return ipc.open_file(pa.BufferReader(chunk)).read_all().num_rows


async def http_units(client: httpx.AsyncClient, url: str, decode, target: int, n_rows: int):
    """(ttfr_s, t_cut_s, rows) for a length-prefixed stream, stopping (and dropping the connection) at `target` rows."""
    t0 = time.perf_counter()
    ttfr, rows, buffer = None, 0, b""
    async with client.stream("GET", url) as resp:
        resp.raise_for_status()
        async for data in resp.aiter_bytes():
            buffer += data
            while len(buffer) >= 8:
                size = int.from_bytes(buffer[:8], "big")
                if len(buffer) < 8 + size:
                    break
                rows += decode(buffer[8:8 + size])
                buffer = buffer[8 + size:]
                if ttfr is None:
                    ttfr = time.perf_counter() - t0
                if rows >= target:
                    return ttfr, time.perf_counter() - t0, rows
    return ttfr, time.perf_counter() - t0, rows


async def http_blob(client: httpx.AsyncClient, url: str, decode):
    t0 = time.perf_counter()
    resp = await client.get(url)
    resp.raise_for_status()
    rows = decode(resp.content)
    elapsed = time.perf_counter() - t0
    return elapsed, elapsed, rows


def flight_fetch(spec: dict, target: int, via_info: bool, total: int):
    t0 = time.perf_counter()
    client = flight.FlightClient(f"grpc+tcp://{SERVER_HOST}:{FLIGHT_PORT}")
    raw = json.dumps(spec).encode()
    try:
        ticket = flight.Ticket(raw)
        if via_info:
            ticket = client.get_flight_info(flight.FlightDescriptor.for_command(raw)).endpoints[0].ticket
        reader = client.do_get(ticket)
        ttfr, rows = None, 0
        while True:
            try:
                chunk = reader.read_chunk()
            except StopIteration:
                return ttfr, time.perf_counter() - t0, rows
            rows += chunk.data.num_rows
            if ttfr is None:
                ttfr = time.perf_counter() - t0
            if rows >= target and rows < total:
                t_cut = time.perf_counter() - t0
                reader.cancel()
                return ttfr, t_cut, rows
    finally:
        client.close()


def flight_action(name: str):
    client = flight.FlightClient(f"grpc+tcp://{SERVER_HOST}:{FLIGHT_PORT}")
    try:
        return json.loads(next(iter(client.do_action(flight.Action(name, b"")))).body.to_pybytes())
    finally:
        client.close()


async def probe_chunks(method: str = "GET") -> int:
    async with httpx.AsyncClient() as client:
        return (await client.request(method, f"{BASE}/probe/chunks")).json()["chunks"]


def local_url(url: str) -> str:
    return url.replace("http://localhost:8000", BASE)


async def run_arm(arm: str, cut: str, session, rid: str, name: str, n_rows: int, n_cols: int):
    """One timed fetch. Returns the record fields after the link has gone quiet."""
    target = target_rows(cut, n_rows)
    call_s = 0.0
    uses_flight = arm.startswith("flight")
    if uses_flight:
        await asyncio.to_thread(flight_action, "reset")
    else:
        await probe_chunks("DELETE")
    await settle()
    rx0 = nic_rx()

    if arm == "json":
        t = time.perf_counter()
        records = await call_large_json(session, n_rows, n_cols, result_id=rid)
        ttfr = t_cut = call_s = time.perf_counter() - t
        rows = len(records)
        rx_cut = nic_rx() - rx0
    elif uses_flight:
        codec = arm.split("_")[1]
        ttfr, t_cut, rows = await asyncio.to_thread(
            flight_fetch, {"table": name, "codec": codec}, target, arm.endswith("_info"), n_rows)
        rx_cut = nic_rx() - rx0
    else:
        t = time.perf_counter()
        if arm == "parquet_stream":
            desc = await call_large_parquet_stream(session, n_rows, n_cols, ROWS_PER_CHUNK, result_id=rid)
        elif arm == "arrow_stream":
            desc = await call_large_arrow_ipc_stream(session, n_rows, n_cols, ROWS_PER_CHUNK, result_id=rid)
        elif arm == "parquet_blob":
            desc = await call_large_parquet_blob(session, n_rows, n_cols, result_id=rid)
        else:
            desc = await call_large_arrow_ipc_blob(session, n_rows, n_cols, result_id=rid)
        call_s = time.perf_counter() - t
        await settle()
        rx0 = nic_rx()
        url = local_url(desc["url"])
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
            if arm == "parquet_stream":
                ttfr, t_cut, rows = await http_units(client, url, _decode_parquet_chunk, target, n_rows)
            elif arm == "arrow_stream":
                ttfr, t_cut, rows = await http_units(client, url, _decode_ipc_chunk, target, n_rows)
            elif arm == "parquet_blob":
                ttfr, t_cut, rows = await http_blob(client, url, _decode_parquet_chunk)
            else:
                ttfr, t_cut, rows = await http_blob(client, url, _decode_ipc_chunk)
            rx_cut = nic_rx() - rx0
    await settle()
    rx_settled = nic_rx() - rx0
    if uses_flight:
        units = (await asyncio.to_thread(flight_action, "stats"))["batches_pulled"]
    elif arm == "json":
        units = math.ceil(n_rows / ROWS_PER_CHUNK)
    else:
        units = await probe_chunks()
    return call_s, ttfr, t_cut, rows, rx_cut, rx_settled, units


def arms_for(args) -> list[tuple[str, str]]:
    plan = [(a, c) for a in ("parquet_stream", "arrow_stream") for c in CUTS]
    plan += [("parquet_blob", "100%"), ("arrow_blob", "100%")]
    plan += [(f"flight_{codec}", c) for codec in FLIGHT_CODECS for c in CUTS]
    plan += [("flight_zstd_info", "first"), ("flight_zstd_info", "100%")]
    if args.json:
        plan.append(("json", "100%"))
    return plan


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/netem_docker/transport.csv")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--tables", default="tpcds,mixed")
    parser.add_argument("--json", action="store_true", help="include the JSON tool (only sensible on fast links)")
    args = parser.parse_args()
    profile = os.environ.get("PROFILE", "unshaped")
    header = ["profile", "table", "arm", "cut", "rep", "call_s", "ttfr_s", "t_cut_s", "rows", "rx_cut", "rx_settled",
              "server_units", "total_units"]
    rows_out = []
    wanted = tables()

    async with connect(f"{BASE}/mcp/mcp", strict_origins=False) as (session, client):
        client.timeout = httpx.Timeout(600.0)
        pings = []
        for _ in range(10):
            t = time.perf_counter()
            await session.send_ping()
            pings.append(time.perf_counter() - t)
        print(f"{profile}: ping rtt median {statistics.median(pings) * 1000:.1f} ms", flush=True)
        for name in args.tables.split(","):
            table = wanted[name]
            buf = io.BytesIO()
            pq.write_table(table, buf, compression="zstd")
            up = await register_materialized(client, buf.getvalue(), base_url=BASE)
            rid, n_rows, n_cols = up["result_id"], up["n_rows"], up["n_cols"]
            plan = arms_for(args)
            total_units = math.ceil(n_rows / ROWS_PER_CHUNK)
            for rep in range(args.reps + 1):  # rep 0 is a warm-up and is not recorded
                order = plan[:]
                random.Random(rep).shuffle(order)
                for arm, cut in order:
                    arm_started = time.perf_counter()
                    try:
                        call_s, ttfr, t_cut, got, rx_cut, rx_settled, units = await run_arm(
                            arm, cut, session, rid, name, n_rows, n_cols)
                    except Exception as exc:  # keep the sweep going; the row records the failure
                        print(f"{profile} {name} {arm} {cut} rep {rep}: FAILED {exc!r}", flush=True)
                        continue
                    if TRACE:
                        print(f"  {arm} {cut}: whole step {time.perf_counter() - arm_started:6.1f} s, descriptor call "
                              f"{call_s:5.2f} s, fetch to the cut {t_cut:6.2f} s", flush=True)
                    if rep == 0:
                        continue
                    rows_out.append([profile, name, arm, cut, rep, round(call_s, 4), round(ttfr or 0, 4),
                                     round(t_cut, 4), got, rx_cut, rx_settled, units, total_units])
                    print(f"{profile:9s} {name:6s} {arm:17s} {cut:5s} rep {rep} ttfr {ttfr or 0:7.3f}s cut {t_cut:7.3f}s "
                          f"rx {rx_cut:>11,d} settled {rx_settled:>11,d} units {units}/{total_units}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows_out)


if __name__ == "__main__":
    asyncio.run(main())
