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

    def link(self, path, who):
        with as_principal(who):
            return security.signed(path)

    def test_another_principal_cannot_read_the_blob_even_with_a_link_signed_for_them(self):
        rid = self.register()
        path = f"/blobs/{rid}.parquet"
        self.assertEqual(self.client.get(self.link(path, "alice"), headers=ALICE).status_code, 200)
        self.assertEqual(self.client.get(self.link(path, "bob"), headers=BOB).status_code, 404)

    def test_a_foreign_handle_looks_like_an_unknown_one(self):
        rid = self.register()
        foreign = self.client.get(self.link(f"/blobs/{rid}.parquet", "bob"), headers=BOB)
        unknown = self.client.get(self.link("/blobs/does-not-exist.parquet", "bob"), headers=BOB)
        self.assertEqual((foreign.status_code, foreign.content), (unknown.status_code, unknown.content))

    def test_derived_handles_inherit_the_owner(self):
        rid = self.register()
        with as_principal("alice"):
            out = S.large_parquet_blob(0, 0, result_id=rid).structured_content
        path = out["url"].replace("http://localhost:8000", "")
        self.assertEqual(self.client.get(path, headers=ALICE).status_code, 200)
        self.assertNotEqual(self.client.get(path, headers=BOB).status_code, 200)

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


class SignedLinks(SecureTestCase):
    def fetch_url(self, who=ALICE):
        rid = self.register(who)
        name = "alice" if who is ALICE else "bob"
        with as_principal(name):
            out = S.large_parquet_blob(0, 0, result_id=rid).structured_content
        return out["url"].replace("http://localhost:8000", "")

    def test_descriptor_urls_are_signed_and_work(self):
        path = self.fetch_url()
        self.assertIn("sig=", path)
        self.assertEqual(self.client.get(path, headers=ALICE).status_code, 200)

    def test_an_unsigned_link_is_refused(self):
        path = self.fetch_url().split("?")[0]
        r = self.client.get(path, headers=ALICE)
        self.assertEqual(r.status_code, 403)
        self.assertIn("missing signature", r.text)

    def test_a_tampered_expiry_is_refused(self):
        path = self.fetch_url()
        base, query = path.split("?")
        exp = int(query.split("exp=")[1].split("&")[0])
        r = self.client.get(path.replace(f"exp={exp}", f"exp={exp + 3600}"), headers=ALICE)
        self.assertEqual(r.status_code, 403)
        self.assertIn("bad signature", r.text)

    def test_an_expired_link_is_refused(self):
        os.environ["TRANSFORM_URL_TTL_S"] = "-1"
        self.addCleanup(os.environ.pop, "TRANSFORM_URL_TTL_S")
        path = self.fetch_url()
        r = self.client.get(path, headers=ALICE)
        self.assertEqual(r.status_code, 403)
        self.assertIn("expired", r.text)

    def test_a_link_is_bound_to_the_principal_it_was_minted_for(self):
        path = self.fetch_url()
        r = self.client.get(path, headers=BOB)
        self.assertEqual(r.status_code, 403)
        self.assertIn("bad signature", r.text)

    def test_a_signature_does_not_transfer_to_another_path(self):
        path = self.fetch_url()
        other = self.fetch_url().split("?")[0]
        r = self.client.get(other + "?" + path.split("?")[1], headers=ALICE)
        self.assertEqual(r.status_code, 403)

    def test_without_any_keys_links_still_expire(self):
        os.environ.pop("MCP_API_KEYS")
        r = self.client.post("/materialized", content=parquet_bytes(), headers={"Content-Type": "application/octet-stream"})
        rid = r.json()["result_id"]
        out = S.large_parquet_blob(0, 0, result_id=rid).structured_content
        url = out["url"].replace("http://localhost:8000", "")
        self.assertEqual(self.client.get(url).status_code, 200)
        self.assertEqual(self.client.get(url.split("?")[0]).status_code, 403)


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
