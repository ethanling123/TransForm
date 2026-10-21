"""Link speed learned from client-reported downloads (FORMAT_LEARN_LINK=1).

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

import format_selector  # noqa: E402
import link_estimator  # noqa: E402
import server_app as S  # noqa: E402


class environ:
    def __init__(self, **values):
        self.values = values

    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in self.values}
        os.environ.update(self.values)

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


class LinkEstimator(unittest.TestCase):
    def setUp(self):
        link_estimator.reset()
        os.environ.pop("FORMAT_LATENCY_NETWORK_MBPS", None)

    def test_no_estimate_until_enough_large_downloads_were_reported(self):
        link_estimator.observe(1_000_000, 1.0)
        link_estimator.observe(1_000_000, 1.0)
        self.assertIsNone(link_estimator.estimate_mbps())
        link_estimator.observe(1_000_000, 1.0)
        self.assertAlmostEqual(link_estimator.estimate_mbps(), 8.0)

    def test_small_downloads_are_ignored_because_round_trips_dominate_them(self):
        for _ in range(5):
            link_estimator.observe(5_000, 0.5)
        self.assertIsNone(link_estimator.estimate_mbps())

    def test_the_median_shrugs_off_one_outlier(self):
        for rate_bytes in (1_000_000, 1_000_000, 1_000_000, 100_000_000):
            link_estimator.observe(rate_bytes, 1.0)
        self.assertAlmostEqual(link_estimator.estimate_mbps(), 8.0)

    def test_the_selector_uses_the_estimate_only_when_enabled_and_nothing_is_configured(self):
        for _ in range(3):
            link_estimator.observe(1_250_000, 1.0)  # 10 Mbit/s
        self.assertEqual(format_selector._get_network_mbps(), 0.0)
        with environ(FORMAT_LEARN_LINK="1"):
            self.assertAlmostEqual(format_selector._get_network_mbps(), 10.0)
            with environ(FORMAT_LATENCY_NETWORK_MBPS="50"):
                self.assertEqual(format_selector._get_network_mbps(), 50.0)

    def test_record_format_outcome_feeds_it_and_reports_the_estimate(self):
        with environ(FORMAT_LEARN_LINK="1"):
            for _ in range(3):
                out = S.record_format_outcome(0, 0, "min_latency", "raw_blob", bytes=2_500_000, latency_s=2.0).structured_content
        self.assertAlmostEqual(out["link_mbps_estimate"], 10.0)


if __name__ == "__main__":
    unittest.main()
