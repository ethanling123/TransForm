"""bird_query_run_inline's JSON fast path returns inline records without allocating a result id.

Run from python-server/:  .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_SCRATCH = tempfile.mkdtemp(prefix="transform-test-")
os.environ.setdefault("TRANSFORM_SCRATCH_DIR", _SCRATCH)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(_SCRATCH)

import shared  # noqa: E402,F401
import server_app as S  # noqa: E402


class InlineJsonPathTest(unittest.TestCase):
    def test_json_choice_returns_inline_records(self):
        db = Path(_SCRATCH) / "tiny.sqlite"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE t (id INTEGER, name TEXT)")
            conn.executemany("INSERT INTO t VALUES (?, ?)", [(1, "a"), (2, "b")])
        with mock.patch.object(S, "_resolve_bird_sqlite_path", return_value=str(db)), \
                mock.patch.object(S, "select_format_with_hints", return_value="json"):
            result = S.bird_query_run_inline("tiny", "SELECT id, name FROM t")
        structured = result.structured_content
        self.assertEqual(structured["chosen_format"], "json")
        self.assertEqual(structured["decode"]["transport"], "inline")


if __name__ == "__main__":
    unittest.main()
