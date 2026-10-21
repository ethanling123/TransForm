"""The experimental inline-in-_meta transport (FORMAT_INLINE_MAX_BYTES).

Run from python-server/:  .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import base64
import hashlib
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
import pyarrow.parquet as pq  # noqa: E402

import shared  # noqa: E402
import server_app as S  # noqa: E402

KEY = S.INLINE_META_KEY
TEXT = b"".join(b'{"id":%d,"event":"click","path":"/item/%d"}\n' % (i, i % 97) for i in range(4000))


class environ:
    def __init__(self, **values):
        self.values = values

    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in self.values}
        os.environ.update(self.values)

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def meta_of(result):
    return (result.meta or {}).get(KEY)


class InlineTransport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = shared.client(S.app)
        n = 3000
        cls.table = pa.table({"id": np.arange(n), "x": np.random.default_rng(0).random(n), "tag": [f"t{i % 9}" for i in range(n)]})
        buf = io.BytesIO()
        pq.write_table(cls.table, buf)
        cls.rid = cls.client.post("/materialized", content=buf.getvalue(),
                                  headers={"Content-Type": "application/octet-stream"}).json()["result_id"]

    def auto(self, rid, target="min_bytes"):
        return S.large_result_auto(0, 0, result_id=rid, optimization_target=target)

    def served(self, structured):
        url = structured["decode"]["url"].replace("http://localhost:8000", "")
        return self.client.get(url).content

    def test_off_by_default_nothing_is_attached(self):
        result = self.auto(self.rid)
        self.assertIsNone(meta_of(result))
        self.assertEqual(result.structured_content["decode"]["transport"], "http_blob")

    def test_small_table_comes_back_in_meta_and_matches_the_blob_route(self):
        with environ(FORMAT_INLINE_MAX_BYTES="1000000"):
            result = self.auto(self.rid)
        structured = result.structured_content
        self.assertEqual(structured["decode"]["transport"], "inline_meta")
        meta = meta_of(result)
        data = base64.b64decode(meta["data"])
        self.assertEqual(hashlib.sha256(data).hexdigest(), meta["sha256"])
        self.assertEqual(data, self.served(structured), "the descriptor URL must still serve the same bytes")
        if structured["chosen_format"] == "parquet_blob":
            self.assertEqual(pq.read_table(io.BytesIO(data)).num_rows, self.table.num_rows)

    def test_a_payload_over_the_cap_stays_a_descriptor(self):
        with environ(FORMAT_INLINE_MAX_BYTES="100"):
            result = self.auto(self.rid)
        self.assertIsNone(meta_of(result))
        self.assertEqual(result.structured_content["decode"]["transport"], "http_blob")

    def test_the_round_trip_saved_must_exceed_the_base64_cost(self):
        with environ(FORMAT_INLINE_MAX_BYTES="1000000", FORMAT_LATENCY_RTT_MS="1", FORMAT_LATENCY_NETWORK_MBPS="1"):
            self.assertFalse(S._should_inline(200_000))  # 0.5 s of extra transfer to save 1 ms
            self.assertTrue(S._should_inline(100))

    def test_unstructured_codec_arm_inlines_the_compressed_bytes(self):
        rid = self.client.post("/materialized-raw", content=TEXT, headers={"Content-Type": "text/plain"}).json()["result_id"]
        with environ(FORMAT_INLINE_MAX_BYTES="1000000"):
            result = self.auto(rid)
        structured = result.structured_content
        self.assertEqual(structured["decode"]["transport"], "inline_meta")
        data = base64.b64decode(meta_of(result)["data"])
        self.assertEqual(data, self.served({"decode": {"url": structured["payload"]["url"]}}) if structured["chosen_format"] != "raw_blob" else TEXT)


if __name__ == "__main__":
    unittest.main()


class RangeRequests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = shared.client(S.app)
        cls.data = bytes(range(256)) * 40
        cls.url = "/raw/" + cls.client.post("/materialized-raw", content=cls.data, headers={"Content-Type": "application/octet-stream"}).json()["result_id"]

    def test_a_plain_get_is_the_whole_payload_and_advertises_ranges(self):
        r = self.client.get(self.url)
        self.assertEqual((r.status_code, r.content, r.headers["accept-ranges"]), (200, self.data, "bytes"))

    def test_a_range_returns_exactly_that_slice(self):
        r = self.client.get(self.url, headers={"Range": "bytes=100-199"})
        self.assertEqual((r.status_code, r.content), (206, self.data[100:200]))
        self.assertEqual(r.headers["content-range"], f"bytes 100-199/{len(self.data)}")

    def test_open_ended_and_suffix_ranges(self):
        self.assertEqual(self.client.get(self.url, headers={"Range": f"bytes={len(self.data) - 10}-"}).content, self.data[-10:])
        self.assertEqual(self.client.get(self.url, headers={"Range": "bytes=-25"}).content, self.data[-25:])

    def test_slices_put_back_together_are_the_original(self):
        n, step = len(self.data), 3000
        parts = [self.client.get(self.url, headers={"Range": f"bytes={a}-{min(a + step, n) - 1}"}).content for a in range(0, n, step)]
        self.assertEqual(b"".join(parts), self.data)

    def test_a_range_past_the_end_is_416(self):
        self.assertEqual(self.client.get(self.url, headers={"Range": f"bytes={len(self.data) + 5}-"}).status_code, 416)


class ConditionalFetch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = shared.client(S.app)
        raw = cls.client.post("/materialized-raw", content=b"payload " * 5000, headers={"Content-Type": "text/plain"}).json()["result_id"]
        n = 2000
        buf = io.BytesIO()
        pq.write_table(pa.table({"id": np.arange(n), "x": np.arange(n) * 1.5}), buf)
        table = cls.client.post("/materialized", content=buf.getvalue(), headers={"Content-Type": "application/octet-stream"}).json()["result_id"]
        cls.table_id = table
        cls.urls = {"raw": f"/raw/{raw}",
                    "parquet": S.large_parquet_blob(0, 0, result_id=table).structured_content["url"].replace("http://localhost:8000", ""),
                    "arrow": S.large_arrow_ipc_blob(0, 0, result_id=table).structured_content["url"].replace("http://localhost:8000", "")}

    def test_every_blob_route_sends_an_etag_and_honours_if_none_match(self):
        for name, url in self.urls.items():
            first = self.client.get(url)
            etag = first.headers["etag"]
            again = self.client.get(url, headers={"If-None-Match": etag})
            self.assertEqual((again.status_code, again.content), (304, b""), name)
            self.assertEqual(again.headers["etag"], etag)

    def test_a_different_etag_gets_the_full_payload(self):
        r = self.client.get(self.urls["raw"], headers={"If-None-Match": '"sha256-0000"'})
        self.assertEqual((r.status_code, len(r.content)), (200, len(b"payload " * 5000)))

    def test_changed_bytes_change_the_etag(self):
        other = self.client.post("/materialized-raw", content=b"other " * 5000, headers={"Content-Type": "text/plain"}).json()["result_id"]
        self.assertNotEqual(self.client.get(self.urls["raw"]).headers["etag"], self.client.get(f"/raw/{other}").headers["etag"])

    def test_the_digest_header_is_only_on_a_full_response(self):
        desc = S.fetch_arm("raw_blob", self.urls["raw"].rsplit("/", 1)[1]).structured_content
        url = desc["data_url"].replace("http://localhost:8000", "")
        full = self.client.get(url)
        self.assertEqual(full.headers["x-content-digest"], hashlib.sha256(full.content).hexdigest())
        part = self.client.get(url, headers={"Range": "bytes=0-9"})
        self.assertNotIn("x-content-digest", part.headers)
        again = self.client.get(url, headers={"If-None-Match": full.headers.get("etag", '"none"')})
        self.assertNotIn("x-content-digest", again.headers)
