"""The chunked arms decode back to exactly the registered rows.

Run from python-server/:  .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

_SCRATCH = tempfile.mkdtemp(prefix="transform-test-")
os.environ.setdefault("TRANSFORM_SCRATCH_DIR", _SCRATCH)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(_SCRATCH)

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.ipc as ipc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import shared  # noqa: E402
import server_app as S  # noqa: E402


def chunks(body: bytes):
    pos = 0
    while pos < len(body):
        size = int.from_bytes(body[pos:pos + 8], "big")
        yield body[pos + 8:pos + 8 + size]
        pos += 8 + size


class StreamArms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = shared.client(S.app)
        rng = np.random.default_rng(1)
        n = 20_000
        cls.table = pa.table({
            "id": np.arange(n),
            "price": rng.random(n),
            "tag": np.array([f"t{i % 17}" for i in range(n)]),
            "when": (1_700_000_000 + np.arange(n)).astype("datetime64[s]"),
        })
        buf = io.BytesIO()
        pq.write_table(cls.table, buf)
        cls.rid = cls.client.post("/materialized", content=buf.getvalue(),
                                  headers={"Content-Type": "application/octet-stream"}).json()["result_id"]

    def fetch(self, descriptor):
        return self.client.get(descriptor["url"].replace("http://localhost:8000", "")).content

    def test_parquet_stream_chunks_rebuild_the_table(self):
        for strategy in ("default", "data_driven"):
            desc = S.large_parquet_stream(0, 0, rows_per_chunk=3000, result_id=self.rid, encoding_strategy=strategy).structured_content
            parts = [pq.read_table(io.BytesIO(c)) for c in chunks(self.fetch(desc))]
            self.assertEqual([p.num_rows for p in parts], [3000] * 6 + [2000])
            joined = pa.concat_tables(parts)
            for name in self.table.column_names:
                self.assertEqual(joined[name].to_pylist(), self.table[name].to_pylist(), f"{strategy}: {name}")

    def test_arrow_ipc_stream_chunks_rebuild_the_table(self):
        desc = S.large_arrow_ipc_stream(0, 0, rows_per_chunk=3000, result_id=self.rid).structured_content
        parts = [ipc.open_file(pa.BufferReader(c)).read_all() for c in chunks(self.fetch(desc))]
        joined = pa.concat_tables(parts)
        self.assertEqual(joined.num_rows, self.table.num_rows)
        for name in self.table.column_names:
            self.assertEqual(joined[name].to_pylist(), self.table[name].to_pylist(), name)


if __name__ == "__main__":
    unittest.main()
