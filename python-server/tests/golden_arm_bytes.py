"""Digests of what every existing arm serves for a fixed fixture.

The golden file was written before the arm registry existed (python tests/golden_arm_bytes.py --write regenerates it).
Parquet and Arrow bytes depend on the pyarrow version; regenerate only after checking that a difference is the library.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import tempfile
from io import BytesIO
from pathlib import Path

_SCRATCH = tempfile.mkdtemp(prefix="transform-golden-")
os.environ.setdefault("TRANSFORM_SCRATCH_DIR", _SCRATCH)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(_SCRATCH)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

GOLDEN = Path(__file__).with_name("golden_arm_bytes.json")


def fixture_parquet() -> bytes:
    rng = np.random.default_rng(7)
    n = 5000
    df = pd.DataFrame({
        "id": np.arange(n),
        "score": rng.normal(size=n).round(3),
        "city": rng.choice(["oslo", "lima", "pune", "kyiv"], n),
        "note": [f"row {i} " + "x" * (i % 17) for i in range(n)],
    })
    buf = BytesIO()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf)
    return buf.getvalue()


def fixture_text() -> bytes:
    return b"".join(b"%d,%d,%s\n" % (i, i * 7, b"abc" * (i % 5)) for i in range(60_000))


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collect(S, client) -> dict:
    out: dict = {}
    reg = client.post("/materialized", content=fixture_parquet(), headers={"Content-Type": "application/octet-stream"})
    rid = reg.json()["result_id"]
    for encoding in ("default", "data_driven"):
        desc = S.large_parquet_blob(0, 0, encoding_strategy=encoding, result_id=rid).structured_content
        out[f"parquet_blob/{encoding}"] = digest(client.get(desc["url"].replace("http://localhost:8000", "")).content)
    for comp in ("none", "lz4", "zstd"):
        desc = S.large_arrow_ipc_blob(0, 0, ipc_compression=comp, result_id=rid).structured_content
        out[f"arrow_ipc_blob/{comp}"] = digest(client.get(desc["url"].replace("http://localhost:8000", "")).content)
    for tool, key in ((S.large_parquet_stream, "parquet_stream"), (S.large_arrow_ipc_stream, "arrow_ipc_stream")):
        desc = tool(0, 0, rows_per_chunk=1000, result_id=rid).structured_content
        out[key] = digest(client.get(desc["url"].replace("http://localhost:8000", "")).content)
    out["json"] = digest(json.dumps(S.large_json(0, 0, result_id=rid), sort_keys=True, default=str).encode())

    text = fixture_text()
    rid = client.post("/materialized-raw", content=text, headers={"Content-Type": "text/csv"}).json()["result_id"]
    out["raw_blob"] = digest(client.get(f"/raw/{rid}").content)
    gz = client.get(f"/raw-gzip/{rid}")
    out["gzip_blob/decompressed"] = digest(gz.content if gz.headers.get("content-encoding") == "gzip" else gzip.decompress(gz.content))
    for route in ("raw-zstd", "raw-brotli"):
        out[route] = digest(client.get(f"/{route}/{rid}").content)
    return out


if __name__ == "__main__":
    import server_app as S

    with TestClient(S.app) as client:
        result = collect(S, client)
    if "--write" in sys.argv:
        GOLDEN.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result, indent=1, sort_keys=True))
