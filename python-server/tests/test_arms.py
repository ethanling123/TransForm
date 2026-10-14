"""Tests for the arm registry, the /data/{handle} route and the registered-only table codec arms.

Run from python-server/:  .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import hashlib
import json
import typing
import unittest

from golden_arm_bytes import GOLDEN, collect, fixture_parquet, fixture_text  # sets up the scratch dir and sys.path

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import arms  # noqa: E402
import format_mab  # noqa: E402
import format_selector  # noqa: E402
import server_app as S  # noqa: E402
import shared  # noqa: E402

LOCAL = "http://localhost:8000"
NEW_TABLE_ARMS = ["parquet_none_blob", "parquet_gzip_blob", "parquet_lz4_blob", "parquet_zstd1_blob",
                  "parquet_zstd3_blob", "parquet_zstd9_blob", "arrow_ipc_lz4_blob", "arrow_ipc_zstd_blob"]
LEGACY = ["json", "parquet_blob", "parquet_stream", "arrow_ipc_blob", "arrow_ipc_stream", "text_inline", "raw_blob",
          "gzip_blob", "zstd_blob", "brotli_blob"]


def path_of(url: str) -> str:
    return url.replace(LOCAL, "")


class Registry(unittest.TestCase):
    def test_selectable_arms_match_the_selector_and_the_bandit(self):
        selectable = set(arms.REGISTRY.names(selectable_only=True))
        self.assertEqual(selectable, set(format_mab.FORMATS))
        self.assertEqual(selectable, set(typing.get_args(format_selector.FormatName)))
        self.assertEqual(selectable, set(LEGACY))

    def test_new_table_arms_are_registered_but_not_selectable(self):
        for name in NEW_TABLE_ARMS:
            spec = arms.REGISTRY.get(name)
            self.assertFalse(spec.selectable, name)
            self.assertEqual((spec.modality, spec.transport), ("table", "blob"))
            self.assertTrue(spec.lossless)

    def test_unknown_arm_raises(self):
        with self.assertRaises(KeyError):
            arms.REGISTRY.get("nope")

    def test_for_modality(self):
        table = {s.name for s in arms.REGISTRY.for_modality("table")}
        self.assertTrue({"json", "parquet_blob", *NEW_TABLE_ARMS} <= table)
        self.assertNotIn("raw_blob", table)

    def test_duplicate_registration_is_rejected(self):
        with self.assertRaises(ValueError):
            arms.REGISTRY.register(arms.REGISTRY.get("json"))

    def test_selector_never_returns_a_non_selectable_arm(self):
        selectable = set(arms.REGISTRY.names(selectable_only=True))
        for target in format_selector.OptimizationTarget:
            for cells in (10, 5_000, 2_000_000):
                hints = {"json_bytes": cells * 8, "parquet_bytes": cells, "arrow_ipc_bytes": cells * 2}
                ctx = format_selector.SelectionContext(n_rows=cells // 4, n_cols=4, target=target, prefer_streaming=False)
                self.assertIn(format_selector.select_format_with_hints(ctx, hints), selectable)


class Encoders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = pq.read_table(pa.BufferReader(fixture_parquet()))

    def test_parquet_codecs_and_levels_round_trip(self):
        sizes = {}
        for codec, level in (("none", None), ("snappy", None), ("gzip", None), ("lz4", None), ("zstd", 1), ("zstd", 9)):
            data = arms.encode_parquet(self.table, codec, level, "default")
            self.assertTrue(pq.read_table(pa.BufferReader(data)).equals(self.table), (codec, level))
            sizes[(codec, level)] = len(data)
        self.assertLess(sizes[("zstd", 9)], sizes[("zstd", 1)])
        self.assertLess(sizes[("zstd", 1)], sizes[("none", None)])

    def test_parquet_encoder_is_the_plain_pyarrow_call(self):
        import io
        buf = io.BytesIO()
        pq.write_table(self.table, buf, compression="zstd", compression_level=3)
        self.assertEqual(arms.encode_parquet(self.table, "zstd", 3, "default"), buf.getvalue())

    def test_parquet_wrapper_in_server_matches_registry_encoder(self):
        for strategy in ("default", "data_driven"):
            self.assertEqual(S._encode_parquet(self.table, "snappy", strategy),
                             arms.encode_parquet(self.table, "snappy", None, strategy))

    def test_arrow_ipc_round_trip(self):
        for codec in ("none", "lz4", "zstd"):
            data = arms.encode_arrow_ipc(self.table, codec)
            self.assertTrue(pa.ipc.open_file(pa.BufferReader(data)).read_all().equals(self.table), codec)

    def test_raw_codec_round_trip(self):
        payload = fixture_text()
        for codec, level in (("zstd", 3), ("brotli", 5), ("gzip", None)):
            packed = arms.encode_raw(payload, codec, level)
            self.assertEqual(pa.Codec(codec).decompress(packed, decompressed_size=len(payload), asbytes=True), payload)


class DataPlane(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = shared.client(S.app)
        cls.table_id = cls.client.post("/materialized", content=fixture_parquet(),
                                       headers={"Content-Type": "application/octet-stream"}).json()["result_id"]
        cls.text = fixture_text()
        cls.text_id = cls.client.post("/materialized-raw", content=cls.text,
                                      headers={"Content-Type": "text/csv"}).json()["result_id"]

    def fetch(self, arm, rid=None):
        out = S.fetch_arm(arm, result_id=rid or self.table_id)
        return out.structured_content

    def test_legacy_golden_bytes_are_unchanged(self):
        self.assertEqual(collect(S, self.client), json.loads(GOLDEN.read_text()))

    def test_every_new_table_arm_serves_a_decodable_payload(self):
        for name in NEW_TABLE_ARMS:
            desc = self.fetch(name)
            self.assertEqual((desc["mode"], desc["format"], desc["modality"]), (name, name, "table"), name)
            response = self.client.get(path_of(desc["data_url"]))
            self.assertEqual(response.status_code, 200, name)
            self.assertEqual(response.headers["x-content-digest"], hashlib.sha256(response.content).hexdigest())
            if name.startswith("parquet"):
                parquet = pq.ParquetFile(pa.BufferReader(response.content))
                expected = {"parquet_none_blob": "UNCOMPRESSED", "parquet_gzip_blob": "GZIP", "parquet_lz4_blob": "LZ4"}
                codec = parquet.metadata.row_group(0).column(0).compression
                self.assertEqual(codec, expected.get(name, "ZSTD"), name)
                self.assertEqual(parquet.read().num_rows, 5000)
            else:
                self.assertEqual(pa.ipc.open_file(pa.BufferReader(response.content)).read_all().num_rows, 5000)

    def test_parquet_levels_are_separate_arms_with_different_sizes(self):
        sizes = {n: len(self.client.get(path_of(self.fetch(n)["data_url"])).content)
                 for n in ("parquet_zstd1_blob", "parquet_zstd3_blob", "parquet_zstd9_blob")}
        self.assertLess(sizes["parquet_zstd9_blob"], sizes["parquet_zstd1_blob"])
        self.assertEqual(len(set(sizes.values())), 3)

    def test_data_route_matches_the_legacy_route_for_every_legacy_blob_arm(self):
        cases = [
            (S.large_parquet_blob(0, 0, result_id=self.table_id).structured_content, None),
            (S.large_arrow_ipc_blob(0, 0, result_id=self.table_id).structured_content, None),
            (S.large_parquet_stream(0, 0, rows_per_chunk=1000, result_id=self.table_id).structured_content, None),
            (S.large_arrow_ipc_stream(0, 0, rows_per_chunk=1000, result_id=self.table_id).structured_content, None),
        ]
        for name in ("raw_blob", "gzip_blob", "zstd_blob", "brotli_blob"):
            cases.append((self.fetch(name, self.text_id), None))
        for desc, _ in cases:
            legacy = self.client.get(path_of(desc["url"]))
            via_data = self.client.get(path_of(desc["data_url"]))
            self.assertEqual(via_data.status_code, 200, desc["mode"])
            if desc["mode"] == "gzip_blob":
                self.assertEqual(S.gzip.decompress(via_data.content) if via_data.headers.get("content-encoding") != "gzip"
                                 else via_data.content, self.text)
            else:
                self.assertEqual(via_data.content, legacy.content, desc["mode"])
            if not desc["mode"].endswith("_stream"):
                self.assertEqual(via_data.headers["x-payload-codec"], str(desc["codec"]))
                if desc["mode"] != "gzip_blob":  # httpx has already undone Content-Encoding for gzip
                    self.assertEqual(via_data.headers["x-content-digest"], hashlib.sha256(via_data.content).hexdigest())

    def test_descriptors_keep_their_old_fields_and_gain_the_new_ones(self):
        desc = S.large_parquet_blob(0, 0, result_id=self.table_id).structured_content
        for old in ("mode", "id", "url", "n_rows", "n_cols", "compression", "encoding_strategy"):
            self.assertIn(old, desc)
        for new in ("modality", "mime_type", "codec", "data_url"):
            self.assertIn(new, desc)
        self.assertEqual(desc["url"].split("/")[3], "blobs")
        self.assertTrue(desc["data_url"].startswith(f"{LOCAL}/data/"))

    def test_size_and_digest_appear_only_when_the_bytes_are_already_cached(self):
        desc = S.large_parquet_blob(0, 0, result_id=self.table_id).structured_content
        body = self.client.get(path_of(desc["data_url"])).content
        self.assertEqual(desc["size"], len(body))
        self.assertEqual(desc["digest"], hashlib.sha256(body).hexdigest())
        cold = self.fetch("parquet_zstd9_blob")
        self.assertNotIn("digest", cold)

    def test_inline_arms_and_unminted_ids_are_not_served_by_data(self):
        self.assertEqual(self.client.get(f"/data/{self.table_id}").status_code, 400)
        self.assertEqual(self.client.get("/data/not-a-handle").status_code, 404)

    def test_fetch_arm_rejects_unknown_wrong_modality_and_inline_arms(self):
        for arm, rid in (("nope", self.table_id), ("raw_blob", self.table_id), ("parquet_zstd3_blob", self.text_id),
                         ("json", self.table_id)):
            with self.assertRaises(ValueError, msg=arm):
                S.fetch_arm(arm, result_id=rid)

    def test_list_arms_describes_the_registry(self):
        listing = S.list_arms().structured_content["arms"]
        by_name = {a["name"]: a for a in listing}
        self.assertEqual(set(by_name), set(arms.REGISTRY.names()))
        self.assertEqual(by_name["parquet_zstd3_blob"]["selectable"], False)
        self.assertEqual(by_name["parquet_zstd3_blob"]["level"], 3)
        self.assertEqual(by_name["parquet_blob"]["selectable"], True)


if __name__ == "__main__":
    unittest.main()
