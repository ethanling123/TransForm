"""
Persisted reference table for format hint estimates and observed outcomes.

This is intentionally simple: a small SQLite DB keyed by a stable JSON key.
It lets the server avoid recomputing expensive encode-size estimates for
repeat requests (especially for materialized result_id).
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


HintsDict = Dict[str, Any]


def _default_db_path() -> Path:
    raw = os.environ.get("FORMAT_HINTS_DB_PATH", "").strip()
    if raw:
        return Path(raw)
    return Path("results/format_hints.sqlite")


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS format_hints (
          key_json TEXT PRIMARY KEY,
          hints_json TEXT NOT NULL,
          updated_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS format_outcomes (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          key_json TEXT NOT NULL,
          outcome_json TEXT NOT NULL,
          created_at REAL NOT NULL
        )
        """
    )
    conn.commit()


def _stable_key_json(key: dict[str, Any]) -> str:
    return json.dumps(key, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class HintStore:
    db_path: Path

    @classmethod
    def default(cls) -> "HintStore":
        return cls(db_path=_default_db_path())

    def get(self, key: dict[str, Any]) -> Optional[HintsDict]:
        key_json = _stable_key_json(key)
        conn = _connect(self.db_path)
        try:
            _ensure_schema(conn)
            cur = conn.execute(
                "SELECT hints_json FROM format_hints WHERE key_json = ?",
                (key_json,),
            )
            row = cur.fetchone()
            if not row:
                return None
            try:
                data = json.loads(row[0])
            except json.JSONDecodeError:
                return None
            return data if isinstance(data, dict) else None
        finally:
            conn.close()

    def upsert(self, key: dict[str, Any], hints: HintsDict) -> None:
        key_json = _stable_key_json(key)
        hints_json = json.dumps(hints, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        conn = _connect(self.db_path)
        try:
            _ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO format_hints(key_json, hints_json, updated_at)
                VALUES(?, ?, ?)
                ON CONFLICT(key_json) DO UPDATE SET
                  hints_json = excluded.hints_json,
                  updated_at = excluded.updated_at
                """,
                (key_json, hints_json, time.time()),
            )
            conn.commit()
        finally:
            conn.close()

    def record_outcome(self, key: dict[str, Any], outcome: dict[str, Any]) -> None:
        """
        Store an observed outcome (bytes/latency/ttfr etc.) so later tooling can
        update the reference table, calibrations, or MAB state.
        """
        key_json = _stable_key_json(key)
        outcome_json = json.dumps(outcome, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        conn = _connect(self.db_path)
        try:
            _ensure_schema(conn)
            conn.execute(
                "INSERT INTO format_outcomes(key_json, outcome_json, created_at) VALUES(?, ?, ?)",
                (key_json, outcome_json, time.time()),
            )
            conn.commit()
        finally:
            conn.close()

