"""The unmodified Python server plus a counter of stream chunks produced, for the early-termination experiment.

server_app is imported as is; its two stream generators are wrapped to count each chunk they yield, and a /probe/chunks
route reports (GET) or clears (DELETE) the count. Run from python-server/ so server_app imports:

    python benchmarks/netem_docker/probe_server.py   (PORT env, default 8000)
"""

from __future__ import annotations

import json
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python-server"))

import server_app  # noqa: E402
import uvicorn  # noqa: E402
from starlette.responses import Response  # noqa: E402
from starlette.routing import Route  # noqa: E402

_lock = threading.Lock()
_chunks = 0


def _counting(generator_fn):
    def wrapper(config):
        global _chunks
        for chunk in generator_fn(config):
            with _lock:
                _chunks += 1
            yield chunk

    return wrapper


server_app._stream_parquet_chunks = _counting(server_app._stream_parquet_chunks)
server_app._stream_arrow_ipc_chunks = _counting(server_app._stream_arrow_ipc_chunks)


async def probe_chunks(request) -> Response:
    global _chunks
    with _lock:
        body = json.dumps({"chunks": _chunks})
        if request.method == "DELETE":
            _chunks = 0
    return Response(body, media_type="application/json")


server_app.app.router.routes.append(Route("/probe/chunks", probe_chunks, methods=["GET", "DELETE"]))

if __name__ == "__main__":
    uvicorn.run(server_app.app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), timeout_keep_alive=600)
