"""Regression tests for the changes made when restoring the Python server.

Run from python-server/:  .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import io
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

# The server opens a hints database relative to the working directory at import time, so run from a temp folder.
_SCRATCH = tempfile.mkdtemp(prefix="transform-test-")
os.environ.setdefault("TRANSFORM_SCRATCH_DIR", _SCRATCH)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(_SCRATCH)

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import server_app as S  # noqa: E402


def make_db(statements):
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp.close()
    con = sqlite3.connect(tmp.name)
    for stmt in statements:
        con.execute(stmt)
    con.commit()
    con.close()
    return Path(tmp.name)


class SqlResultCleaning(unittest.TestCase):
    def test_duplicate_column_names_are_renamed(self):
        db = make_db([
            "CREATE TABLE a (name TEXT, loc TEXT)", "CREATE TABLE b (name TEXT)",
            "INSERT INTO a VALUES ('x', 'l')", "INSERT INTO b VALUES ('y')",
        ])
        df = S._execute_sqlite_query_to_df(db, "SELECT a.name, a.loc, b.name FROM a, b", 100)
        self.assertEqual(list(df.columns), ["name", "loc", "name_1"])
        pa.Table.from_pandas(df, preserve_index=False)  # Arrow rejects duplicates, so this must not raise

    def test_mixed_type_column_becomes_strings(self):
        db = make_db(["CREATE TABLE t (v)", "INSERT INTO t VALUES (1)", "INSERT INTO t VALUES ('abc')"])
        df = S._execute_sqlite_query_to_df(db, "SELECT v FROM t", 100)
        self.assertEqual(sorted(df["v"]), ["1", "abc"])
        pa.Table.from_pandas(df, preserve_index=False)

    def test_invalid_utf8_is_replaced(self):
        db = make_db(["CREATE TABLE t (v TEXT)", "INSERT INTO t VALUES (CAST(x'41ff42' AS TEXT))"])
        df = S._execute_sqlite_query_to_df(db, "SELECT v FROM t", 100)
        self.assertEqual(df["v"][0], "A�B")


class DerivedConfigCaches(unittest.TestCase):
    def setUp(self):
        table = pa.table({"a": list(range(1000)), "b": [float(i) for i in range(1000)]})
        buf = io.BytesIO()
        pq.write_table(table, buf)
        self.src = S.ResultConfig(n_rows=1000, n_cols=2, payload_kind="tabular")
        S._populate_materialized_caches(
            self.src, result_id="src-test", df=table.to_pandas(), table=table
        )
        S.RESULT_REGISTRY["src-test"] = self.src

    def tearDown(self):
        for key in [k for k in S.RESULT_REGISTRY if k == "src-test" or k in S._DERIVED_BY_SOURCE.get("src-test", [])]:
            S.RESULT_REGISTRY.pop(key, None)
        S._DERIVED_BY_SOURCE.pop("src-test", None)

    def _derived_id(self, result):
        return result.structured_content["id"]

    def test_parquet_blob_descriptor_shares_primed_cache(self):
        result = S.large_parquet_blob(1000, 2, result_id="src-test")
        derived = S.RESULT_REGISTRY[self._derived_id(result)]
        self.assertIsNotNone(derived.cached_parquet_blob_bytes)
        self.assertIs(derived.cached_parquet_blob_bytes, self.src.cached_parquet_blob_bytes)

    def test_arrow_blob_descriptor_shares_primed_cache(self):
        result = S.large_arrow_ipc_blob(1000, 2, result_id="src-test")
        derived = S.RESULT_REGISTRY[self._derived_id(result)]
        self.assertIs(derived.cached_arrow_ipc_blob_bytes, self.src.cached_arrow_ipc_blob_bytes)

    def test_evicting_the_source_releases_derived_caches(self):
        result = S.large_parquet_blob(1000, 2, result_id="src-test")
        derived = S.RESULT_REGISTRY[self._derived_id(result)]
        S._payload_cache_clear_entry("src-test")
        self.assertIsNone(self.src.cached_parquet_blob_bytes)
        self.assertIsNone(derived.cached_parquet_blob_bytes)
        self.assertIsNone(derived.cached_dataframe)


class ToolSurface(unittest.TestCase):
    def test_describe_and_fetch_result_matches_large_result_auto(self):
        a = S.describe_and_fetch_result(n_rows=10, n_cols=3)
        b = S.large_result_auto(n_rows=10, n_cols=3)
        self.assertEqual(a.structured_content["chosen_format"], b.structured_content["chosen_format"])


if __name__ == "__main__":
    unittest.main()
