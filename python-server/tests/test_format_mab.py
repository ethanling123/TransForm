"""The bandit may only explore and exploit among arms that apply to the payload at hand.

Run from python-server/:  .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

_SCRATCH = tempfile.mkdtemp(prefix="transform-test-")
os.environ.setdefault("TRANSFORM_SCRATCH_DIR", _SCRATCH)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(_SCRATCH)

import format_mab  # noqa: E402
import format_selector  # noqa: E402

TABLE_HINTS = {"json_bytes": 5_000_000, "parquet_bytes": 400_000, "arrow_ipc_bytes": 500_000}
UNSTRUCTURED_HINTS = {"raw_bytes": 3_000_000, "gzip_bytes": 900_000}
CONTEXT = format_selector.SelectionContext(n_rows=100_000, n_cols=20)


def state_favouring(arm: str) -> dict:
    """A bandit state in which `arm` has the best reward and every other arm has been tried once."""
    target = CONTEXT.target.value
    return {target: {fmt: {"sum_reward": -10.0 if fmt != arm else -0.01, "count": 1.0} for fmt in format_mab.FORMATS}}


class FeasibleArmsTest(unittest.TestCase):
    def test_exploration_stays_among_table_arms(self):
        random.seed(0)
        state = state_favouring("parquet_blob")
        picks = {format_mab.select_format_with_mab(CONTEXT, TABLE_HINTS, state, epsilon=1.0) for _ in range(300)}
        self.assertLessEqual(picks, set(format_mab.TABLE_FORMATS))

    def test_exploration_stays_among_unstructured_arms(self):
        random.seed(0)
        state = state_favouring("raw_blob")
        picks = {format_mab.select_format_with_mab(CONTEXT, UNSTRUCTURED_HINTS, state, epsilon=1.0) for _ in range(300)}
        self.assertLessEqual(picks, set(format_mab.UNSTRUCTURED_FORMATS))

    def test_greedy_ignores_a_better_arm_that_does_not_apply(self):
        state = state_favouring("text_inline")
        pick = format_mab.select_format_with_mab(CONTEXT, TABLE_HINTS, state, epsilon=0.0)
        self.assertIn(pick, format_mab.TABLE_FORMATS)

    def test_greedy_still_picks_the_best_applicable_arm(self):
        state = state_favouring("arrow_ipc_blob")
        self.assertEqual(format_mab.select_format_with_mab(CONTEXT, TABLE_HINTS, state, epsilon=0.0), "arrow_ipc_blob")

    def test_no_history_for_applicable_arms_falls_back_to_the_selector(self):
        state = {CONTEXT.target.value: {"text_inline": {"sum_reward": -1.0, "count": 3.0}}}
        pick = format_mab.select_format_with_mab(CONTEXT, TABLE_HINTS, state, epsilon=0.0)
        self.assertEqual(pick, format_selector.select_format_with_hints(CONTEXT, TABLE_HINTS))

    def test_every_arm_is_in_one_of_the_two_sets(self):
        self.assertEqual(set(format_mab.TABLE_FORMATS) | set(format_mab.UNSTRUCTURED_FORMATS), set(format_mab.FORMATS))


if __name__ == "__main__":
    unittest.main()
