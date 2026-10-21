"""Tests for the experimental cost-aware unstructured selector (FORMAT_COST_AWARE=1).

Run from python-server/:  .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

_SCRATCH = tempfile.mkdtemp(prefix="transform-test-")
os.environ.setdefault("TRANSFORM_SCRATCH_DIR", _SCRATCH)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(_SCRATCH)

import cost_model  # noqa: E402
import shared  # noqa: E402
import server_app as S  # noqa: E402
from format_selector import OptimizationTarget, SelectionContext, select_format_with_hints  # noqa: E402

TEXT = b"".join(b'{"id":%d,"name":"item-%d","tags":["a","b"]}\n' % (i, i % 97) for i in range(30_000))
NOISE = os.urandom(1_500_000)


def choose(hints, target=OptimizationTarget.MIN_LATENCY):
    return select_format_with_hints(SelectionContext(n_rows=0, n_cols=0, target=target), hints)


class environ:
    def __init__(self, **values):
        self.values = values

    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in self.values}
        os.environ.update(self.values)

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


class CostModel(unittest.TestCase):
    def test_codec_levels_match_the_arms_the_server_serves(self):
        self.assertEqual(S.RAW_BYTE_CODECS["zstd_blob"][:2], ("zstd", 3))
        self.assertEqual(S.RAW_BYTE_CODECS["brotli_blob"][:2], ("brotli", 5))

    def test_calibration_has_every_codec_with_positive_speeds(self):
        speeds = cost_model.calibrate()
        self.assertEqual(set(speeds), set(cost_model.CODECS))
        self.assertTrue(all(c > 0 and d > 0 for c, d in speeds.values()))

    def test_fast_link_skips_sampling_entirely(self):
        def read_sample():
            raise AssertionError("sampled on a link too fast for any codec to pay")

        hints = cost_model.estimate_hints(read_sample, 10_000_000, mbps=1_000_000)
        self.assertEqual(hints["stage"], "break_even_skip")
        self.assertIsNone(hints["zstd_bytes"])

    def test_incompressible_data_stops_after_the_zstd_sample(self):
        hints = cost_model.estimate_hints(lambda: NOISE[:cost_model.SAMPLE_BYTES], len(NOISE), mbps=10)
        self.assertEqual(hints["stage"], "zstd_not_worth")
        self.assertIsNone(hints["gzip_bytes"])
        self.assertIsNone(hints["brotli_bytes"])

    def test_a_fast_link_only_samples_zstd_because_nothing_slower_can_win(self):
        hints = cost_model.estimate_hints(lambda: TEXT[:cost_model.SAMPLE_BYTES], len(TEXT), mbps=1000)
        self.assertEqual(hints["stage"], "zstd_dominates")
        self.assertLess(hints["zstd_bytes"], len(TEXT))
        self.assertIsNone(hints["gzip_bytes"])
        self.assertIsNone(hints["brotli_bytes"])

    def test_compressible_data_on_a_slow_link_samples_every_codec(self):
        # 1 Mbit/s keeps zstd's transfer time well above gzip's and brotli's extra compute on any machine
        hints = cost_model.estimate_hints(lambda: TEXT[:cost_model.SAMPLE_BYTES], len(TEXT), mbps=1)
        self.assertEqual(hints["stage"], "sampled")
        for key in ("zstd_bytes", "gzip_bytes", "brotli_bytes"):
            self.assertLess(hints[key], len(TEXT))


class SelectorWithComputeTime(unittest.TestCase):
    def hints(self, compute_s):
        return {"raw_bytes": 1_000_000, "zstd_bytes": 400_000, "brotli_bytes": 300_000, "compute_s": compute_s}

    def test_compute_time_can_outweigh_the_byte_saving(self):
        with environ(FORMAT_LATENCY_NETWORK_MBPS="1000"):
            # 600 KB saved is 4.8 ms at 1 Gbit/s; 50 ms of compute is not worth it
            self.assertEqual(choose(self.hints({"zstd_blob": 0.05, "brotli_blob": 0.2})), "raw_blob")

    def test_slow_link_pays_for_the_compute(self):
        with environ(FORMAT_LATENCY_NETWORK_MBPS="10"):
            self.assertEqual(choose(self.hints({"zstd_blob": 0.05, "brotli_blob": 0.1})), "brotli_blob")

    def test_min_bytes_ignores_compute_time(self):
        with environ(FORMAT_LATENCY_NETWORK_MBPS="1000"):
            self.assertEqual(choose(self.hints({"zstd_blob": 5.0, "brotli_blob": 5.0}), OptimizationTarget.MIN_BYTES), "brotli_blob")


class ServerHints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = shared.client(S.app)

    def register(self, payload: bytes, content_type="text/plain") -> str:
        response = self.client.post("/materialized-raw", content=payload, headers={"Content-Type": content_type})
        self.assertEqual(response.status_code, 201)
        return response.json()["result_id"]

    def test_cost_aware_hints_compress_and_write_nothing(self):
        rid = self.register(TEXT)
        cfg = S.RESULT_REGISTRY[rid]
        hints = S._unstructured_hints(cfg, rid, cost_aware_mbps=10)
        self.assertTrue(hints["estimated"])
        self.assertEqual(cfg.raw_codec_paths, {})
        self.assertIsNone(cfg.raw_gzip_path)

    def test_legacy_hints_still_compress_exactly(self):
        rid = self.register(TEXT)
        cfg = S.RESULT_REGISTRY[rid]
        hints = S._unstructured_hints(cfg, rid)
        self.assertNotIn("estimated", hints)
        self.assertEqual(hints["zstd_bytes"], len(S._raw_codec_blob(cfg, rid, "zstd_blob")))
        self.assertTrue(cfg.raw_codec_paths)

    def call_auto(self, rid, target):
        return S.large_result_auto(0, 0, result_id=rid, optimization_target=target).structured_content

    def test_auto_with_flag_on_a_fast_link_goes_raw_without_compressing(self):
        rid = self.register(TEXT)
        with environ(FORMAT_COST_AWARE="1", FORMAT_LATENCY_NETWORK_MBPS="1000000"):
            out = self.call_auto(rid, "min_latency")
        self.assertEqual(out["chosen_format"], "raw_blob")
        self.assertEqual(S.RESULT_REGISTRY[rid].raw_codec_paths, {})

    def test_auto_with_flag_on_a_slow_link_picks_a_codec_for_text(self):
        rid = self.register(TEXT)
        with environ(FORMAT_COST_AWARE="1", FORMAT_LATENCY_NETWORK_MBPS="5"):
            out = self.call_auto(rid, "min_latency")
        self.assertIn(out["chosen_format"], ("zstd_blob", "gzip_blob", "brotli_blob"))

    def test_auto_min_bytes_keeps_exact_sizes_with_the_flag_on(self):
        rid = self.register(TEXT)
        with environ(FORMAT_COST_AWARE="1", FORMAT_LATENCY_NETWORK_MBPS="1000000"):
            out = self.call_auto(rid, "min_bytes")
        self.assertIn(out["chosen_format"], ("zstd_blob", "gzip_blob", "brotli_blob"))
        self.assertTrue(S.RESULT_REGISTRY[rid].raw_codec_paths)

    def test_flag_off_behaves_as_before_even_on_a_fast_link(self):
        rid = self.register(TEXT)
        with environ(FORMAT_LATENCY_NETWORK_MBPS="1000000"):
            self.call_auto(rid, "min_latency")
        self.assertTrue(S.RESULT_REGISTRY[rid].raw_codec_paths)

    def test_incompressible_payload_goes_raw_on_a_slow_link(self):
        rid = self.register(NOISE, "application/octet-stream")
        with environ(FORMAT_COST_AWARE="1", FORMAT_LATENCY_NETWORK_MBPS="5"):
            out = self.call_auto(rid, "min_latency")
        self.assertEqual(out["chosen_format"], "raw_blob")
        self.assertEqual(S.RESULT_REGISTRY[rid].raw_codec_paths, {})


if __name__ == "__main__":
    unittest.main()


class TableArmChoice(unittest.TestCase):
    """cost_model.choose_table_arm with stand-in encoders, so the outcome does not depend on this machine's speed."""

    def sample(self):
        import pyarrow as pa
        return pa.table({"x": list(range(1000))})

    def candidates(self, sizes, cached_default=True, delay=None):
        import time

        def make(size, wait):
            def encode(table):
                time.sleep(wait)
                return b"x" * size
            return encode
        names = list(sizes)
        return [(n, make(sizes[n], (delay or {}).get(n, 0.0)), lambda b: None, cached_default and i == 0) for i, n in enumerate(names)]

    def test_a_much_smaller_arm_wins_on_a_slow_link(self):
        best, _ = cost_model.choose_table_arm(self.sample(), 1000, self.candidates({"default": 1_000_000, "small": 400_000}), mbps=10)
        self.assertEqual(best, "small")

    def test_a_marginal_gain_keeps_the_default(self):
        best, _ = cost_model.choose_table_arm(self.sample(), 1000, self.candidates({"default": 1_000_000, "small": 980_000}), mbps=10)
        self.assertEqual(best, "default")

    def test_encode_time_counts_against_an_arm_that_is_not_cached(self):
        # 0.05 s of sample encode time at a 1000x scale is 50 s, far more than the 0.5 s saved on the wire
        best, report = cost_model.choose_table_arm(
            self.sample(), 1_000_000, self.candidates({"default": 1_000, "slow": 100}, delay={"slow": 0.05}), mbps=1)
        self.assertEqual(best, "default")
        self.assertGreater(report["slow"]["encode_s"], 40)


class TableCodecInServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import io
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        cls.client = shared.client(S.app)
        rng = np.random.default_rng(0)
        n = 60_000
        table = pa.table({"id": np.arange(n), "x": rng.random(n), "tag": [f"category-{i % 40}" for i in range(n)]})
        buf = io.BytesIO()
        pq.write_table(table, buf)
        cls.rid = cls.client.post("/materialized", content=buf.getvalue(), headers={"Content-Type": "application/octet-stream"}).json()["result_id"]
        cls.rows = n

    def auto(self, target="min_latency"):
        return S.large_result_auto(0, 0, result_id=self.rid, optimization_target=target).structured_content

    def test_flag_off_keeps_the_default_arm(self):
        with environ(FORMAT_LATENCY_NETWORK_MBPS="5"):
            self.assertNotIn("arm", self.auto()["decode"])

    def test_flag_on_returns_a_decodable_blob_whatever_arm_it_picks(self):
        import io
        import pyarrow as pa
        import pyarrow.parquet as pq
        with environ(FORMAT_COST_AWARE="1", FORMAT_LATENCY_NETWORK_MBPS="5"):
            out = self.auto()
        data = self.client.get(out["decode"]["url"].replace("http://localhost:8000", "")).content
        if out["decode"]["encoding"] == "parquet":
            rows = pq.read_table(io.BytesIO(data)).num_rows
        else:
            rows = pa.ipc.open_file(pa.BufferReader(data)).read_all().num_rows
        self.assertEqual(rows, self.rows)

    def test_min_bytes_never_swaps_the_arm(self):
        with environ(FORMAT_COST_AWARE="1", FORMAT_LATENCY_NETWORK_MBPS="5"):
            self.assertNotIn("arm", self.auto("min_bytes")["decode"])
