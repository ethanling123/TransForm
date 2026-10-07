"""TCP proxy emulating a network link: one-way delay, bandwidth cap and packet loss.

Delay is applied per direction (as `tc netem` does on a loopback interface).
Loss is approximated: each lost ~1460-byte packet costs one extra round trip.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import random

CHUNK = 16384
MTU = 1460


async def pipe(reader, writer, delay_s, mbps, loss):
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    link_free = 0.0
    last_deliver = 0.0

    async def sender():
        while True:
            item = await queue.get()
            if item is None:
                break
            deliver_at, data = item
            wait = deliver_at - loop.time()
            if wait > 0:
                await asyncio.sleep(wait)
            writer.write(data)
            await writer.drain()
        writer.close()

    task = asyncio.create_task(sender())
    try:
        while True:
            data = await reader.read(CHUNK)
            if not data:
                break
            start = max(loop.time(), link_free)
            link_free = start + len(data) * 8 / (mbps * 1e6)
            lost = sum(random.random() < loss for _ in range(math.ceil(len(data) / MTU))) if loss else 0
            last_deliver = max(last_deliver, link_free + delay_s + lost * 2 * delay_s)
            queue.put_nowait((last_deliver, data))
    finally:
        queue.put_nowait(None)
        await task


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", type=int, required=True)
    parser.add_argument("--target", type=int, default=8000)
    parser.add_argument("--delay-ms", type=float, required=True)
    parser.add_argument("--mbps", type=float, required=True)
    parser.add_argument("--loss", type=float, default=0.0)
    args = parser.parse_args()
    delay_s = args.delay_ms / 1000

    async def handle(client_r, client_w):
        await asyncio.sleep(2 * delay_s)  # TCP handshake round trip
        try:
            server_r, server_w = await asyncio.open_connection("127.0.0.1", args.target)
        except OSError:
            client_w.close()
            return
        await asyncio.gather(
            pipe(client_r, server_w, delay_s, args.mbps, args.loss),
            pipe(server_r, client_w, delay_s, args.mbps, args.loss),
            return_exceptions=True,
        )

    server = await asyncio.start_server(handle, "127.0.0.1", args.listen)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
