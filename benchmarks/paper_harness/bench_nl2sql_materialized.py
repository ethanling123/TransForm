"""
End-to-end NL2SQL materialized benchmark runner.

Executes SQL from BIRD / Spider / WikiSQL against local databases,
registers the result DataFrames as Parquet on the MCP server, then
measures JSON / Parquet-blob / Parquet-stream delivery for each result.

Usage (from project root, with server running):

    uvicorn server_app:app --reload
    python bench_nl2sql_materialized.py --dataset bird --data-dir data/datasets/bird/dev

Results are appended to results/nl2sql_materialized.jsonl (use --overwrite for a fresh file).

After a run, summarize with: python summarize_nl2sql_materialized.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from client.mcp_client import (
    DEFAULT_MCP_URL,
    connect,
    call_large_json,
    call_large_parquet_blob,
    call_large_parquet_stream,
    call_large_arrow_ipc_blob,
    call_large_arrow_ipc_stream,
    call_describe_result_formats,
    call_large_result_auto,
    call_record_format_outcome,
    fetch_blob,
    fetch_stream_chunks,
    register_materialized,
)
from format_selector import (
    OptimizationTarget,
    SelectionContext,
    get_default_target,
    select_format,
    select_format_with_hints,
)

log = logging.getLogger(__name__)

DEFAULT_RESULTS_PATH = Path("results/nl2sql_materialized.jsonl")
MAX_RESULT_ROWS = 500_000
STREAM_CHUNK_SIZES = [4096, 16384, 65536]


# ---------------------------------------------------------------------------
# Per-query result record
# ---------------------------------------------------------------------------

@dataclass
class QueryBenchRecord:
    dataset: str
    question_id: str
    db_id: str
    sql: str
    n_rows: int
    n_cols: int
    result_id: str
    recommended_format: str
    sql_exec_s: Optional[float] = None
    register_s: Optional[float] = None
    describe_result_formats_s: Optional[float] = None
    json_end_to_end_s: Optional[float] = None
    json_response_bytes: Optional[int] = None
    parquet_blob_end_to_end_s: Optional[float] = None
    parquet_blob_bytes: Optional[int] = None
    parquet_stream_end_to_end_s: Optional[float] = None
    parquet_stream_time_to_first_rows_s: Optional[float] = None
    parquet_stream_bytes: Optional[int] = None
    parquet_stream_rows_per_chunk: Optional[int] = None
    arrow_ipc_blob_end_to_end_s: Optional[float] = None
    arrow_ipc_blob_bytes: Optional[int] = None
    arrow_ipc_stream_end_to_end_s: Optional[float] = None
    arrow_ipc_stream_time_to_first_rows_s: Optional[float] = None
    arrow_ipc_stream_bytes: Optional[int] = None
    arrow_ipc_stream_rows_per_chunk: Optional[int] = None
    recommended_end_to_end_s: Optional[float] = None
    recommended_bytes: Optional[int] = None
    server_auto_chosen_format: Optional[str] = None
    server_auto_end_to_end_s: Optional[float] = None
    server_auto_time_to_first_rows_s: Optional[float] = None
    server_auto_bytes: Optional[int] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# SQL execution helpers (pluggable per dataset)
# ---------------------------------------------------------------------------

def bird_sql_for_sqlite(sql: str) -> str:
    """BIRD gold SQL uses MySQL-style backticks; SQLite expects double-quoted identifiers."""
    return sql.replace("`", '"')


def resolve_bird_sqlite_path(data_dir: Path, db_id: str) -> Optional[Path]:
    """
    Locate <db_id>.sqlite for BIRD. Checks, in order:
    - BIRD_SQLITE_ROOT/dev_databases/<db_id>/<db_id>.sqlite (root = MINIDEV-style folder)
    - data_dir/dev_databases/... and data_dir/databases/...
    - data_dir.parent/minidev/MINIDEV/dev_databases/... (official mini-dev zip layout)
    """
    name = f"{db_id}.sqlite"
    roots: List[Path] = []
    env = os.environ.get("BIRD_SQLITE_ROOT", "").strip()
    if env:
        roots.append(Path(env) / "dev_databases" / db_id / name)
    roots.extend(
        [
            data_dir / "dev_databases" / db_id / name,
            data_dir / "databases" / db_id / name,
            data_dir.parent / "minidev" / "MINIDEV" / "dev_databases" / db_id / name,
        ]
    )
    for p in roots:
        if p.is_file():
            return p
    return None


def execute_sql_sqlite(db_path: str, sql: str, max_rows: int = MAX_RESULT_ROWS) -> pd.DataFrame:
    """Execute *sql* against a SQLite database file and return a DataFrame."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()
    if len(df) > max_rows:
        df = df.iloc[:max_rows]
    return df


def execute_sql_duckdb(db_path: str, sql: str, max_rows: int = MAX_RESULT_ROWS) -> pd.DataFrame:
    """Execute *sql* against a DuckDB database and return a DataFrame."""
    import duckdb
    conn = duckdb.connect(db_path)
    try:
        df = conn.execute(sql).fetchdf()
    finally:
        conn.close()
    if len(df) > max_rows:
        df = df.iloc[:max_rows]
    return df


# ---------------------------------------------------------------------------
# DataFrame → Parquet bytes for registration
# ---------------------------------------------------------------------------

def dedupe_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make column labels unique so Arrow/Parquet registration succeeds
    (e.g. BIRD queries that produce repeated `name` columns).
    """
    seen: Dict[str, int] = {}
    new_cols: List[str] = []
    for c in df.columns:
        base = str(c)
        n = seen.get(base, 0)
        if n == 0:
            new_cols.append(base)
        else:
            new_cols.append(f"{base}__dup{n}")
        seen[base] = n + 1
    out = df.copy()
    out.columns = new_cols
    return out


def df_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    df = dedupe_duplicate_columns(df)
    table = pa.Table.from_pandas(df, preserve_index=False)
    buf = BytesIO()
    pq.write_table(table, buf, compression="snappy")
    return buf.getvalue()


def recommended_path_metrics(metrics: Dict[str, Any]) -> tuple[Optional[float], Optional[int]]:
    """Map recommended_format to measured latency and bytes for that mode."""
    fmt = metrics.get("recommended_format") or ""
    if fmt == "json":
        return metrics.get("json_end_to_end_s"), metrics.get("json_response_bytes")
    if fmt == "parquet_blob":
        return metrics.get("parquet_blob_end_to_end_s"), metrics.get("parquet_blob_bytes")
    if fmt == "parquet_stream":
        return metrics.get("parquet_stream_end_to_end_s"), metrics.get("parquet_stream_bytes")
    if fmt == "arrow_ipc_blob":
        return metrics.get("arrow_ipc_blob_end_to_end_s"), metrics.get("arrow_ipc_blob_bytes")
    if fmt == "arrow_ipc_stream":
        return metrics.get("arrow_ipc_stream_end_to_end_s"), metrics.get("arrow_ipc_stream_bytes")
    return None, None


# ---------------------------------------------------------------------------
# Benchmark a single materialized result
# ---------------------------------------------------------------------------

async def bench_one_result(
    session: Any,
    http: httpx.AsyncClient,
    result_id: str,
    n_rows: int,
    n_cols: int,
    target: OptimizationTarget,
) -> Dict[str, Any]:
    """Run all three format benchmarks for a single registered result."""
    metrics: Dict[str, Any] = {}

    # ---- hints ----
    t_desc0 = time.perf_counter()
    hints = await call_describe_result_formats(
        session,
        n_rows,
        n_cols,
        result_id=result_id,
        optimization_target=target.value,
        prefer_streaming=False,
    )
    metrics["describe_result_formats_s"] = time.perf_counter() - t_desc0
    ctx = SelectionContext(n_rows=n_rows, n_cols=n_cols, target=target)
    recommended = select_format_with_hints(ctx, hints) if hints else select_format(ctx)
    metrics["recommended_format"] = recommended

    # ---- JSON ----
    try:
        t0 = time.perf_counter()
        structured = await call_large_json(session, n_rows, n_cols, result_id=result_id)
        t1 = time.perf_counter()
        raw = json.dumps(structured)
        metrics["json_end_to_end_s"] = t1 - t0
        metrics["json_response_bytes"] = len(raw.encode("utf-8"))
    except Exception as exc:
        log.warning("JSON case failed for %s: %s", result_id, exc)
        metrics["json_end_to_end_s"] = None
        metrics["json_response_bytes"] = None

    # ---- Parquet blob ----
    try:
        t0 = time.perf_counter()
        desc = await call_large_parquet_blob(
            session, n_rows, n_cols, result_id=result_id,
        )
        t1 = time.perf_counter()
        data = await fetch_blob(http, desc["url"])
        t2 = time.perf_counter()
        metrics["parquet_blob_end_to_end_s"] = t2 - t0
        metrics["parquet_blob_bytes"] = len(data)
    except Exception as exc:
        log.warning("Parquet blob case failed for %s: %s", result_id, exc)

    # ---- Parquet stream (first chunk-size config) ----
    rpc = STREAM_CHUNK_SIZES[0]
    try:
        t0 = time.perf_counter()
        desc = await call_large_parquet_stream(
            session, n_rows, n_cols, rows_per_chunk=rpc, result_id=result_id,
        )
        t1 = time.perf_counter()
        url = desc["url"]
        bytes_read = 0
        time_to_first: Optional[float] = None
        async for chunk in fetch_stream_chunks(http, url):
            bytes_read += 8 + len(chunk)
            if time_to_first is None:
                time_to_first = time.perf_counter() - t1
        t2 = time.perf_counter()
        metrics["parquet_stream_end_to_end_s"] = t2 - t0
        metrics["parquet_stream_time_to_first_rows_s"] = time_to_first
        metrics["parquet_stream_bytes"] = bytes_read
        metrics["parquet_stream_rows_per_chunk"] = rpc
    except Exception as exc:
        log.warning("Parquet stream case failed for %s: %s", result_id, exc)

    # ---- Arrow IPC blob ----
    try:
        t0 = time.perf_counter()
        desc = await call_large_arrow_ipc_blob(
            session, n_rows, n_cols, result_id=result_id,
        )
        t1 = time.perf_counter()
        data = await fetch_blob(http, desc["url"])
        t2 = time.perf_counter()
        metrics["arrow_ipc_blob_end_to_end_s"] = t2 - t0
        metrics["arrow_ipc_blob_bytes"] = len(data)
    except Exception as exc:
        log.warning("Arrow IPC blob case failed for %s: %s", result_id, exc)

    # ---- Arrow IPC stream (same chunk size as Parquet stream) ----
    try:
        t0 = time.perf_counter()
        desc = await call_large_arrow_ipc_stream(
            session, n_rows, n_cols, rows_per_chunk=rpc, result_id=result_id,
        )
        t1 = time.perf_counter()
        url = desc["url"]
        bytes_read = 0
        time_to_first: Optional[float] = None
        async for chunk in fetch_stream_chunks(http, url):
            bytes_read += 8 + len(chunk)
            if time_to_first is None:
                time_to_first = time.perf_counter() - t1
        t2 = time.perf_counter()
        metrics["arrow_ipc_stream_end_to_end_s"] = t2 - t0
        metrics["arrow_ipc_stream_time_to_first_rows_s"] = time_to_first
        metrics["arrow_ipc_stream_bytes"] = bytes_read
        metrics["arrow_ipc_stream_rows_per_chunk"] = rpc
    except Exception as exc:
        log.warning("Arrow IPC stream case failed for %s: %s", result_id, exc)

    rec_lat, rec_bytes = recommended_path_metrics(metrics)
    metrics["recommended_end_to_end_s"] = rec_lat
    metrics["recommended_bytes"] = rec_bytes

    # ---- Server-side one-shot selection ----
    try:
        t0 = time.perf_counter()
        auto = await call_large_result_auto(
            session,
            n_rows,
            n_cols,
            rows_per_chunk=STREAM_CHUNK_SIZES[0],
            result_id=result_id,
            optimization_target=target.value,
            prefer_streaming=False,
            use_mab=False,
        )
        chosen = str(auto.get("chosen_format") or "")
        metrics["server_auto_chosen_format"] = chosen

        payload = auto.get("payload") if isinstance(auto, dict) else None
        decode = auto.get("decode") if isinstance(auto, dict) else None
        bytes_read = None
        time_to_first = None
        if isinstance(payload, dict) and payload.get("kind") == "json":
            raw = json.dumps(payload.get("records"))
            bytes_read = len(raw.encode("utf-8"))
        elif isinstance(payload, dict) and payload.get("kind") == "text":
            text = payload.get("text") or ""
            bytes_read = len(str(text).encode("utf-8"))
        elif isinstance(decode, dict) and isinstance(decode.get("url"), str):
            url = decode["url"]
            if decode.get("transport") == "http_length_prefixed_stream":
                t1 = time.perf_counter()
                total = 0
                async for chunk in fetch_stream_chunks(http, url):
                    total += 8 + len(chunk)
                    if time_to_first is None:
                        time_to_first = time.perf_counter() - t1
                bytes_read = total
            else:
                data = await fetch_blob(http, url)
                bytes_read = len(data)

        t2 = time.perf_counter()
        metrics["server_auto_end_to_end_s"] = t2 - t0
        metrics["server_auto_time_to_first_rows_s"] = time_to_first
        metrics["server_auto_bytes"] = bytes_read

        # Optionally update server-side MAB state from this measurement.
        if chosen and bytes_read is not None:
            await call_record_format_outcome(
                session,
                n_rows,
                n_cols,
                optimization_target=target.value,
                format_used=chosen,
                bytes=int(bytes_read),
                latency_s=float(metrics["server_auto_end_to_end_s"]),
                time_to_first_rows_s=float(time_to_first) if time_to_first is not None else None,
            )
    except Exception as exc:
        log.warning("Server auto selection failed for %s: %s", result_id, exc)
        metrics["server_auto_chosen_format"] = None
        metrics["server_auto_end_to_end_s"] = None
        metrics["server_auto_time_to_first_rows_s"] = None
        metrics["server_auto_bytes"] = None

    return metrics


# ---------------------------------------------------------------------------
# Query iterators per dataset (used by the main loop)
# ---------------------------------------------------------------------------

@dataclass
class NL2SQLQuery:
    dataset: str
    question_id: str
    db_id: str
    sql: str
    db_path: str
    engine: str  # "sqlite" | "duckdb"


def iter_bird_queries(
    data_dir: Path,
    *,
    max_queries: int = 200,
    questions_file: str = "dev.json",
) -> Iterator[NL2SQLQuery]:
    """Yield NL2SQL queries from BIRD dev (or mini_dev_sqlite.json) with resolvable .sqlite files."""
    qpath = Path(questions_file)
    if not qpath.is_absolute():
        qpath = (data_dir / questions_file).resolve()
    if not qpath.is_file():
        log.error("BIRD questions file not found: %s", qpath)
        return
    with open(qpath, encoding="utf-8") as f:
        items = json.load(f)
    count = 0
    for item in items:
        if count >= max_queries:
            break
        db_id = item.get("db_id", "")
        sql = item.get("SQL", item.get("sql", ""))
        if not sql or not db_id:
            continue
        resolved = resolve_bird_sqlite_path(data_dir, db_id)
        if resolved is None:
            continue
        qid = item.get("question_id", str(count))
        yield NL2SQLQuery(
            dataset="bird",
            question_id=str(qid),
            db_id=db_id,
            sql=sql,
            db_path=str(resolved),
            engine="sqlite",
        )
        count += 1


def iter_spider_queries(
    data_dir: Path, *, max_queries: int = 200,
) -> Iterator[NL2SQLQuery]:
    """Yield NL2SQL queries from Spider 1.0."""
    for candidate in ["dev.json", "train_spider.json"]:
        p = data_dir / candidate
        if p.exists():
            dev_json = p
            break
    else:
        log.error("Spider JSON not found in %s", data_dir)
        return
    with open(dev_json) as f:
        items = json.load(f)
    db_dir = data_dir / "database"
    count = 0
    for item in items:
        if count >= max_queries:
            break
        db_id = item.get("db_id", "")
        sql = item.get("query", "")
        if not sql or not db_id:
            continue
        db_path = db_dir / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            continue
        yield NL2SQLQuery(
            dataset="spider",
            question_id=str(count),
            db_id=db_id,
            sql=sql,
            db_path=str(db_path),
            engine="sqlite",
        )
        count += 1


def iter_wikisql_queries(
    data_dir: Path, *, max_queries: int = 200,
) -> Iterator[NL2SQLQuery]:
    """Yield NL2SQL queries from WikiSQL."""
    import sqlite3

    for split in ["dev", "test", "train"]:
        tables_file = data_dir / f"{split}.tables.jsonl"
        sql_file = data_dir / f"{split}.jsonl"
        if not tables_file.exists() or not sql_file.exists():
            continue

        tables: Dict[str, Dict] = {}
        with open(tables_file) as f:
            for line in f:
                t = json.loads(line)
                tables[t["id"]] = t

        count = 0
        with open(sql_file) as f:
            for line in f:
                if count >= max_queries:
                    return
                item = json.loads(line)
                table_id = item.get("table_id", "")
                tbl = tables.get(table_id)
                if tbl is None:
                    continue

                sql_obj = item.get("sql", {})
                if not isinstance(sql_obj, dict):
                    continue

                cols = tbl.get("header", [])
                rows_data = tbl.get("rows", [])
                if not cols or not rows_data:
                    continue

                db_path = data_dir / "wikisql_tmp" / f"{table_id}.sqlite"
                db_path.parent.mkdir(parents=True, exist_ok=True)
                if not db_path.exists():
                    _build_wikisql_sqlite(db_path, table_id, cols, rows_data)

                flat_sql = _wikisql_sql_to_string(sql_obj, cols, table_id)
                if not flat_sql:
                    continue

                yield NL2SQLQuery(
                    dataset="wikisql",
                    question_id=str(count),
                    db_id=table_id,
                    sql=flat_sql,
                    db_path=str(db_path),
                    engine="sqlite",
                )
                count += 1


def _build_wikisql_sqlite(
    db_path: Path, table_id: str, cols: List[str], rows: List[List],
) -> None:
    """Create a single-table SQLite DB from WikiSQL table data."""
    import sqlite3
    safe_cols = [f'"{c}"' for c in cols]
    create_sql = f"CREATE TABLE IF NOT EXISTS \"{table_id}\" ({', '.join(f'{c} TEXT' for c in safe_cols)})"
    placeholders = ", ".join(["?"] * len(cols))
    insert_sql = f"INSERT INTO \"{table_id}\" VALUES ({placeholders})"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(create_sql)
        conn.executemany(insert_sql, rows)
        conn.commit()
    finally:
        conn.close()


def _wikisql_sql_to_string(
    sql_obj: Dict, cols: List[str], table_id: str,
) -> Optional[str]:
    """Convert WikiSQL's structured SQL dict to a flat SQL string."""
    sel_idx = sql_obj.get("sel")
    agg_idx = sql_obj.get("agg", 0)
    conds = sql_obj.get("conds", {})

    if sel_idx is None or sel_idx >= len(cols):
        return None

    agg_ops = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
    agg = agg_ops[agg_idx] if 0 <= agg_idx < len(agg_ops) else ""
    col_name = f'"{cols[sel_idx]}"'
    select_clause = f"{agg}({col_name})" if agg else col_name

    where_parts: List[str] = []
    op_map = ["=", ">", "<", ">=", "<=", "!="]
    if isinstance(conds, list):
        for triplet in conds:
            if not isinstance(triplet, (list, tuple)) or len(triplet) < 3:
                continue
            ci, oi, val = int(triplet[0]), int(triplet[1]), triplet[2]
            if ci >= len(cols):
                continue
            op = op_map[oi] if 0 <= oi < len(op_map) else "="
            safe_val = str(val).replace("'", "''")
            where_parts.append(f'"{cols[ci]}" {op} \'{safe_val}\'')
    elif isinstance(conds, dict):
        cond_cols = conds.get("column_index", [])
        cond_ops_idx = conds.get("operator_index", [])
        cond_vals = conds.get("condition", [])
        for ci, oi, val in zip(cond_cols, cond_ops_idx, cond_vals):
            if ci >= len(cols):
                continue
            op = op_map[oi] if 0 <= oi < len(op_map) else "="
            safe_val = str(val).replace("'", "''")
            where_parts.append(f'"{cols[ci]}" {op} \'{safe_val}\'')

    sql = f'SELECT {select_clause} FROM "{table_id}"'
    if where_parts:
        sql += " WHERE " + " AND ".join(where_parts)
    return sql


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

async def run_materialized_bench(
    dataset: str,
    data_dir: Path,
    *,
    mcp_url: str = DEFAULT_MCP_URL,
    server_url: str = "http://localhost:8000",
    max_queries: int = 200,
    results_path: Path = DEFAULT_RESULTS_PATH,
    overwrite: bool = False,
    bird_questions_file: str = "dev.json",
) -> None:
    iterators = {
        "bird": iter_bird_queries,
        "spider": iter_spider_queries,
        "wikisql": iter_wikisql_queries,
    }
    iter_fn = iterators.get(dataset)
    if iter_fn is None:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(iterators)}")

    target = get_default_target()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and results_path.exists():
        results_path.unlink()

    async with connect(base_url=mcp_url) as (session, http):
        if dataset == "bird":
            queries = list(
                iter_bird_queries(
                    data_dir,
                    max_queries=max_queries,
                    questions_file=bird_questions_file,
                )
            )
        else:
            queries = list(iter_fn(data_dir, max_queries=max_queries))
        log.info("Loaded %d queries for dataset=%s", len(queries), dataset)
        if dataset == "bird" and not queries:
            log.error(
                "No BIRD queries loaded. Put <db_id>.sqlite under %s/dev_databases/, set "
                "BIRD_SQLITE_ROOT to a MINIDEV folder, or run: bash scripts/fetch_bird_minidev.sh",
                data_dir,
            )

        for i, q in enumerate(queries):
            log.info(
                "[%d/%d] dataset=%s db=%s qid=%s",
                i + 1, len(queries), q.dataset, q.db_id, q.question_id,
            )

            # Execute SQL locally
            sql_exec_s: Optional[float] = None
            try:
                t_sql = time.perf_counter()
                sql_run = (
                    bird_sql_for_sqlite(q.sql) if q.dataset == "bird" else q.sql
                )
                if q.engine == "duckdb":
                    df = execute_sql_duckdb(q.db_path, sql_run)
                else:
                    df = execute_sql_sqlite(q.db_path, sql_run)
                sql_exec_s = time.perf_counter() - t_sql
            except Exception as exc:
                log.warning("SQL execution failed for qid=%s: %s", q.question_id, exc)
                rec = QueryBenchRecord(
                    dataset=q.dataset, question_id=q.question_id,
                    db_id=q.db_id, sql=q.sql, n_rows=0, n_cols=0,
                    result_id="", recommended_format="",
                    sql_exec_s=sql_exec_s,
                    error=f"sql_exec: {exc}",
                )
                _append_record(rec, results_path)
                continue

            if df.empty:
                log.info("Empty result for qid=%s, skipping", q.question_id)
                continue

            n_rows, n_cols = df.shape

            # Register materialized Parquet on the server
            register_s: Optional[float] = None
            try:
                t_reg = time.perf_counter()
                pq_bytes = df_to_parquet_bytes(df)
                reg = await register_materialized(http, pq_bytes, base_url=server_url)
                register_s = time.perf_counter() - t_reg
                result_id = reg["result_id"]
            except Exception as exc:
                log.warning("Registration failed for qid=%s: %s", q.question_id, exc)
                rec = QueryBenchRecord(
                    dataset=q.dataset, question_id=q.question_id,
                    db_id=q.db_id, sql=q.sql, n_rows=n_rows, n_cols=n_cols,
                    result_id="", recommended_format="",
                    sql_exec_s=sql_exec_s,
                    register_s=register_s,
                    error=f"register: {exc}",
                )
                _append_record(rec, results_path)
                continue

            # Benchmark all three formats
            try:
                metrics = await bench_one_result(
                    session, http, result_id, n_rows, n_cols, target,
                )
            except Exception as exc:
                log.warning("Bench failed for qid=%s: %s", q.question_id, exc)
                metrics = {"error": str(exc), "recommended_format": ""}

            rec = QueryBenchRecord(
                dataset=q.dataset,
                question_id=q.question_id,
                db_id=q.db_id,
                sql=q.sql,
                n_rows=n_rows,
                n_cols=n_cols,
                result_id=result_id,
                sql_exec_s=sql_exec_s,
                register_s=register_s,
                **metrics,
            )
            _append_record(rec, results_path)
            log.info(
                "  → rows=%d cols=%d json=%s blob=%s stream=%s",
                n_rows, n_cols,
                _fmt_ms(metrics.get("json_end_to_end_s")),
                _fmt_ms(metrics.get("parquet_blob_end_to_end_s")),
                _fmt_ms(metrics.get("parquet_stream_end_to_end_s")),
            )


def _fmt_ms(v: Optional[float]) -> str:
    return f"{v*1000:.1f}ms" if v is not None else "N/A"


def _append_record(rec: QueryBenchRecord, path: Path) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(asdict(rec)) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="NL2SQL materialized benchmark")
    parser.add_argument("--dataset", required=True, choices=["bird", "spider", "wikisql"])
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument("--mcp-url", default=DEFAULT_MCP_URL)
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete results file before running (fresh run)",
    )
    parser.add_argument(
        "--bird-questions",
        default="dev.json",
        metavar="FILE",
        help="BIRD questions JSON relative to --data-dir, or absolute path (default: dev.json)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    asyncio.run(
        run_materialized_bench(
            dataset=args.dataset,
            data_dir=args.data_dir,
            mcp_url=args.mcp_url,
            server_url=args.server_url,
            max_queries=args.max_queries,
            results_path=args.results,
            overwrite=args.overwrite,
            bird_questions_file=args.bird_questions,
        )
    )


if __name__ == "__main__":
    main()
