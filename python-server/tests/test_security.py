"""Tests for the secure data plane (TRANSFORM_SECURE=1). Run from python-server/:  .venv/bin/python -m unittest discover -s tests -v"""

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

import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import security  # noqa: E402
import server_app as S  # noqa: E402
import shared  # noqa: E402

KEYS = "alice:key-alice,bob:key-bob"
ALICE = {"Authorization": "Bearer key-alice"}
BOB = {"Authorization": "Bearer key-bob"}


def parquet_bytes(n=50) -> bytes:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(pd.DataFrame({"x": range(n)}), preserve_index=False), buf)
    return buf.getvalue()


class as_principal:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.token = security.current_principal.set(self.name)

    def __exit__(self, *exc):
        security.current_principal.reset(self.token)


class SecureTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = shared.client(S.app)

    def setUp(self):
        saved = {k: os.environ.get(k) for k in ("TRANSFORM_SECURE", "MCP_API_KEYS")}
        os.environ.update(TRANSFORM_SECURE="1", MCP_API_KEYS=KEYS)
        self.addCleanup(self.restore, saved)

    @staticmethod
    def restore(saved):
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

    def register(self, who=ALICE):
        r = self.client.post("/materialized", content=parquet_bytes(), headers={**who, "Content-Type": "application/octet-stream"})
        self.assertEqual(r.status_code, 201, r.text)
        return r.json()["result_id"]


class Principals(SecureTestCase):
    def test_unknown_key_is_rejected(self):
        r = self.client.post("/materialized", content=parquet_bytes(), headers={"Authorization": "Bearer nope"})
        self.assertEqual(r.status_code, 401)

    def test_owner_is_stamped_from_the_key(self):
        self.assertEqual(S.RESULT_REGISTRY.raw(self.register()).owner, "alice")

    def test_another_principal_cannot_read_the_blob(self):
        rid = self.register()
        self.assertEqual(self.client.get(f"/blobs/{rid}.parquet", headers=ALICE).status_code, 200)
        self.assertEqual(self.client.get(f"/blobs/{rid}.parquet", headers=BOB).status_code, 404)

    def test_a_foreign_handle_looks_like_an_unknown_one(self):
        rid = self.register()
        foreign = self.client.get(f"/blobs/{rid}.parquet", headers=BOB)
        unknown = self.client.get("/blobs/does-not-exist.parquet", headers=BOB)
        self.assertEqual((foreign.status_code, foreign.content), (unknown.status_code, unknown.content))

    def test_derived_handles_inherit_the_owner(self):
        rid = self.register()
        with as_principal("alice"):
            out = S.large_parquet_blob(0, 0, result_id=rid).structured_content
        path = out["url"].replace("http://localhost:8000", "")
        self.assertEqual(self.client.get(path, headers=ALICE).status_code, 200)
        self.assertEqual(self.client.get(path, headers=BOB).status_code, 404)

    def test_tools_do_not_resolve_a_foreign_handle(self):
        rid = self.register()
        with as_principal("bob"), self.assertRaises(Exception):
            S.large_json(0, 0, result_id=rid)

    def test_tool_calls_over_mcp_see_the_callers_principal(self):
        rid = self.register()
        call = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "large_json", "arguments": {"n_rows": 0, "n_cols": 0, "result_id": rid}}}
        mcp_headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        own = self.client.post("/mcp/mcp", json=call, headers={**ALICE, **mcp_headers})
        other = self.client.post("/mcp/mcp", json=call, headers={**BOB, **mcp_headers})
        self.assertFalse(own.json()["result"].get("isError"), own.text)
        self.assertTrue(other.json()["result"].get("isError"), other.text)


class FlagOff(unittest.TestCase):
    def test_registry_behaves_like_a_dict_when_the_flag_is_off(self):
        os.environ.pop("TRANSFORM_SECURE", None)
        cfg = S.ResultConfig(n_rows=1, n_cols=1)
        S.RESULT_REGISTRY["flag-off-test"] = cfg
        self.assertIsNone(cfg.owner)
        with as_principal("someone-else"):
            self.assertIs(S.RESULT_REGISTRY.get("flag-off-test"), cfg)


if __name__ == "__main__":
    unittest.main()
