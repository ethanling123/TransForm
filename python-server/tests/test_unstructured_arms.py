"""Tests for the unstructured payload arms: raw, gzip, zstd, brotli and inline text.

Run from python-server/:  .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import gzip
import os
import sys
import tempfile
import unittest
from pathlib import Path

_SCRATCH = tempfile.mkdtemp(prefix="transform-test-")
os.environ.setdefault("TRANSFORM_SCRATCH_DIR", _SCRATCH)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(_SCRATCH)

import pyarrow as pa  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

import server_app as S  # noqa: E402
from format_selector import OptimizationTarget, SelectionContext, select_format_with_hints  # noqa: E402


def choose(hints, target=OptimizationTarget.MIN_BYTES):
    return select_format_with_hints(SelectionContext(n_rows=0, n_cols=0, target=target, prefer_streaming=False), hints)


class SelectorArms(unittest.TestCase):
    def test_smallest_arm_wins(self):
        hints = {"raw_bytes": 100_000, "gzip_bytes": 30_000, "zstd_bytes": 28_000, "brotli_bytes": 25_000}
        self.assertEqual(choose(hints), "brotli_blob")

    def test_ties_prefer_zstd_over_gzip_over_brotli(self):
        hints = {"raw_bytes": 100_000, "gzip_bytes": 30_000, "zstd_bytes": 30_000, "brotli_bytes": 30_000}
        self.assertEqual(choose(hints), "zstd_blob")

    def test_raw_when_nothing_compresses(self):
        self.assertEqual(choose({"raw_bytes": 100_000}), "raw_blob")

    def test_tiny_text_stays_inline(self):
        hints = {"raw_bytes": 2000, "text_inline_bytes": 2000, "gzip_bytes": 600, "zstd_bytes": 500, "brotli_bytes": 450}
        self.assertEqual(choose(hints), "text_inline")

    def test_network_mode_picks_least_transfer(self):
        os.environ["FORMAT_LATENCY_NETWORK_MBPS"] = "10"
        try:
            hints = {"raw_bytes": 900_000, "gzip_bytes": 300_000, "zstd_bytes": 250_000}
            self.assertEqual(choose(hints, OptimizationTarget.MIN_LATENCY), "zstd_blob")
        finally:
            del os.environ["FORMAT_LATENCY_NETWORK_MBPS"]


class ServerArms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(S.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def register(self, payload: bytes, content_type="text/plain") -> str:
        response = self.client.post("/materialized-raw", content=payload, headers={"Content-Type": content_type})
        self.assertEqual(response.status_code, 201)
        return response.json()["result_id"]

    def test_gzip_hint_exists_on_a_fresh_result(self):
        rid = self.register(b"abcdefgh" * 400_000)
        hints = S._unstructured_hints(S.RESULT_REGISTRY[rid], rid)
        self.assertIsNotNone(hints["gzip_bytes"])
        self.assertLess(hints["zstd_bytes"], hints["raw_bytes"])
        self.assertLess(hints["brotli_bytes"], hints["raw_bytes"])

    def test_compressible_payload_uses_a_codec_arm_and_round_trips(self):
        payload = (b"row,value\n" + b"".join(b"%d,%d\n" % (i, i * 7) for i in range(300_000)))
        rid = self.register(payload)
        result = S.large_result_auto(0, 0, result_id=rid, optimization_target="min_bytes")
        structured = result.structured_content
        self.assertIn(structured["chosen_format"], ("zstd_blob", "brotli_blob"))
        decode = structured["decode"]
        response = self.client.get(decode["url"].replace("http://localhost:8000", ""))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-payload-codec"], decode["codec"])
        self.assertEqual(int(response.headers["x-uncompressed-bytes"]), decode["uncompressed_bytes"])
        restored = pa.Codec(decode["codec"]).decompress(response.content, decompressed_size=len(payload), asbytes=True)
        self.assertEqual(restored, payload)
        self.assertLess(len(response.content), len(payload) / 3)

    def test_each_codec_endpoint_round_trips(self):
        payload = (b"the quick brown fox " * 100_000)
        rid = self.register(payload)
        for route, codec in (("raw-zstd", "zstd"), ("raw-brotli", "brotli")):
            response = self.client.get(f"/{route}/{rid}")
            self.assertEqual(response.status_code, 200)
            restored = pa.Codec(codec).decompress(response.content, decompressed_size=len(payload), asbytes=True)
            self.assertEqual(restored, payload)
        gz = self.client.get(f"/raw-gzip/{rid}")
        self.assertEqual(gzip.decompress(gz.content) if gz.headers.get("content-encoding") != "gzip" else gz.content, payload)

    def test_already_compressed_payload_skips_the_codecs(self):
        payload = os.urandom(2 * 1024 * 1024)
        rid = self.register(payload, "application/octet-stream")
        hints = S._unstructured_hints(S.RESULT_REGISTRY[rid], rid)
        self.assertTrue(hints["already_compressed"])
        self.assertIsNone(hints["zstd_bytes"])
        result = S.large_result_auto(0, 0, result_id=rid, optimization_target="min_bytes")
        self.assertEqual(result.structured_content["chosen_format"], "raw_blob")

    def test_describe_lists_the_new_arms(self):
        rid = self.register(b"abcdefgh" * 400_000)
        formats = S.describe_result_formats(0, 0, result_id=rid).structured_content["formats"]
        for arm in ("raw_blob", "gzip_blob", "zstd_blob", "brotli_blob", "text_inline"):
            self.assertIn(arm, formats)


if __name__ == "__main__":
    unittest.main()
