"""Unstructured payloads through the server: raw, gzip, zstd and brotli blobs, inline text, and the selector.

Payloads are real files: the M3-Bench images, the synthesized speech clips, and text/JSON/CSV documents.
For every payload and strategy a fresh result is uploaded (upload time excluded) so each arm pays its own
compression cost cold; the selector row also pays for computing the size hints. Every fetch is decoded and
checked against the original bytes. Forced arms include one MCP ping standing in for the tool call that would
return their URL, so they pay the same control-plane round trip as the selector's tool call.

Uploads go straight to the server (--server-port); tool calls and downloads go through --port, which can be the
network-emulation proxy.

    .venv/bin/python -m benchmarks.sweeps.payloads --out results/payloads/localhost.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sqlite3
import time
from pathlib import Path

import httpx
import pyarrow as pa

from client.mcp_client import connect
from client.verify import open_payload

MIME = {".png": "image/png", ".wav": "audio/wav", ".json": "application/json", ".csv": "text/csv", ".md": "text/markdown",
        ".txt": "text/plain", ".java": "text/x-java"}
FORCED = {"raw": "raw", "gzip": "raw-gzip", "zstd": "raw-zstd", "brotli": "raw-brotli"}
ARM = {"raw": "raw_blob", "gzip": "gzip_blob", "zstd": "zstd_blob", "brotli": "brotli_blob"}
CODEC_OF_ARM = {"zstd_blob": "zstd", "brotli_blob": "brotli"}


def text_payloads(bird_root: str):
    """(name, bytes, mime) for the text documents; anything missing on this machine is skipped."""
    out = []
    for label, path in (("m3bench annotations (json)", "data/m3bench/test_mcp_GT.json"),
                        ("m3bench fuzzy annotations (json)", "data/m3bench/test_mcp_fuzzy.json"),
                        ("bird dev questions (json)", f"{bird_root}/dev.json"),
                        ("spider dev questions (json)", "data/spider_data/dev.json"),
                        ("report (markdown)", "REPORT.md")):
        if Path(path).exists():
            out.append((label, Path(path).read_bytes(), MIME[Path(path).suffix]))
    sources = sorted(Path("java-server/src/main/java").glob("**/*.java"))
    if sources:
        out.append(("java sources (text)", b"".join(p.read_bytes() for p in sources), "text/plain"))
    dev = Path(bird_root) / "dev.json"
    if dev.exists():
        query = next(q for q in json.loads(dev.read_text()) if str(q["question_id"]) == "384")
        con = sqlite3.connect(f"{bird_root}/dev_databases/{query['db_id']}/{query['db_id']}.sqlite")
        cur = con.execute(query["SQL"].replace("`", '"'))
        names = [d[0] for d in cur.description]
        rows = cur.fetchall()
        con.close()
        import io
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(names)
        writer.writerows(rows)
        out.append((f"bird-384 result ({len(rows)} rows, csv)", buf.getvalue().encode(), "text/csv"))
        out.append((f"bird-384 result ({len(rows)} rows, json)",
                    json.dumps([dict(zip(names, r)) for r in rows]).encode(), "application/json"))
    return out


def all_payloads(args):
    for p in sorted(Path(args.images).glob("*.png"))[:: args.image_stride]:
        yield f"image {p.name}", "image", p.read_bytes(), "image/png"
    for p in sorted(Path(args.audio).glob("*.wav")):
        yield f"audio {p.name}", "audio", p.read_bytes(), "audio/wav"
    for name, data, mime in text_payloads(args.bird_root):
        yield name, "text", data, mime


def via(url: str, server_port: int, port: int) -> str:
    return url.replace(f"localhost:{server_port}", f"localhost:{port}").replace(f"127.0.0.1:{server_port}", f"localhost:{port}")


def decode_blob(resp: httpx.Response, codec: str | None) -> bytes:
    """Undo the arm's compression. httpx already undoes Content-Encoding: gzip."""
    if codec in ("zstd", "brotli"):
        size = int(resp.headers["x-uncompressed-bytes"])
        return pa.Codec(codec).decompress(resp.content, decompressed_size=size, asbytes=True)
    return resp.content


async def upload(client, server_port, data, mime) -> str:
    resp = await client.post(f"http://localhost:{server_port}/materialized-raw", content=data, headers={"Content-Type": mime})
    resp.raise_for_status()
    return resp.json()["result_id"]


async def run_forced(session, client, args, name, data, mime, strategy):
    rid = await upload(client, args.server_port, data, mime)
    t = time.perf_counter()
    desc = (await session.call_tool("fetch_arm", arguments={"arm": ARM[strategy], "result_id": rid})).structuredContent
    resp = await client.get(via(desc["url"], args.server_port, args.port))
    resp.raise_for_status()
    decoded = decode_blob(resp, strategy if strategy in ("zstd", "brotli") else None)
    return strategy, resp.num_bytes_downloaded, time.perf_counter() - t, "", decoded == data


async def run_selector(session, client, args, name, data, mime, target):
    rid = await upload(client, args.server_port, data, mime)
    t = time.perf_counter()
    result = await session.call_tool("large_result_auto", arguments={"n_rows": 0, "n_cols": 0, "result_id": rid,
                                                                      "optimization_target": target})
    select_s = time.perf_counter() - t
    out = result.structuredContent
    chosen, decode = out["chosen_format"], out["decode"]
    if decode["transport"] == "inline":
        text = open_payload(out["payload"])
        wire, decoded = len(text.encode()), text.encode()
    else:
        resp = await client.get(via(decode["url"], args.server_port, args.port))
        resp.raise_for_status()
        wire = resp.num_bytes_downloaded
        decoded = decode_blob(resp, decode.get("codec"))
    return f"selector {target}", wire, time.perf_counter() - t, select_s, chosen, decoded == data


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="port for tool calls and downloads (proxy or server)")
    parser.add_argument("--server-port", type=int, default=8000, help="port uploads go to (always the server)")
    parser.add_argument("--images", default="data/m3bench/images")
    parser.add_argument("--audio", default="data/audio")
    parser.add_argument("--bird-root", default="data/dev_20240627")
    parser.add_argument("--image-stride", type=int, default=1)
    parser.add_argument("--out", default="results/payloads/localhost.csv")
    args = parser.parse_args()

    rows = []
    async with connect(f"http://localhost:{args.port}/mcp/mcp") as (session, client):
        client.timeout = httpx.Timeout(900.0)
        for name, kind, data, mime in all_payloads(args):
            for strategy in FORCED:
                arm, wire, latency, _, ok = await run_forced(session, client, args, name, data, mime, strategy)
                rows.append([name, kind, len(data), arm, arm, wire, latency, "", ok])
            for target in ("min_bytes", "min_latency"):
                label, wire, latency, select_s, chosen, ok = await run_selector(session, client, args, name, data, mime, target)
                rows.append([name, kind, len(data), label, chosen, wire, latency, select_s, ok])
            print(f"{name}: {len(data) / 1e6:.2f} MB done", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["payload", "kind", "raw_bytes", "strategy", "arm", "wire_bytes", "latency_s", "select_s", "ok"])
        w.writerows(rows)
    bad = [r for r in rows if not r[-1]]
    print(f"done: {len(rows)} rows, {len(bad)} round-trip mismatches")


if __name__ == "__main__":
    asyncio.run(main())
