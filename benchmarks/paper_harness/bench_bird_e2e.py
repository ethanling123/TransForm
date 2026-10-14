"""
BIRD transport benchmark: SQLite → register → MCP baseline vs enhanced.

**Core evaluation (default):** fixed SQL per question — **gold** from the questions
JSON, or **frozen** SQL from an earlier run (JSONL keyed by question_id + db_id).
This isolates transport + format selection + compression from NL2SQL quality.

**Secondary (optional):** `--sql-source ollama` runs Ollama NL2SQL for an
“end-to-end agent” flavor (small subset / appendix), not required for core numbers.

Compares:
  Baseline: large_json only (no describe_result_formats).
  Enhanced: describe_result_formats + heuristic select_format_with_hints + one fetch.
  Server: one MCP call to large_result_auto (server-side selection + inline JSON or blob/stream descriptor), then optional HTTP fetch for non-inline payloads.

Optional: LLM summary (`--with-summary`) uses Ollama when enabled.

Examples:

    # Core: gold SQL (default)
    uvicorn server_app:app --host 127.0.0.1 --port 8000
    python bench_bird_e2e.py --sql-source gold --max-queries 500 --overwrite

    # Core: frozen SQL from a JSONL artifact
    python bench_bird_e2e.py --sql-source jsonl --frozen-sql results/my_pred.sql.jsonl

    # Secondary: model-generated SQL
    python bench_bird_e2e.py --sql-source ollama --max-queries 30

Then: python summarize_bird_e2e.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import httpx
import pandas as pd

from bench_nl2sql_materialized import (
    bird_sql_for_sqlite,
    df_to_parquet_bytes,
    execute_sql_sqlite,
    resolve_bird_sqlite_path,
)
from client.llm_client import OLLAMA_MODEL, chat, get_llm_backend
from client.mcp_client import (
    DEFAULT_MCP_URL,
    connect,
    call_describe_result_formats,
    call_large_json,
    call_large_parquet_blob,
    call_large_parquet_stream,
    call_large_arrow_ipc_blob,
    call_large_arrow_ipc_stream,
    call_large_result_auto,
    call_bird_query_run_inline,
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


def _rewrite_server_url(url: str, server_url: str) -> str:
    """Map localhost blob URLs to --server-url (e.g. Docker http://server:8000)."""
    if not url or not server_url:
        return url
    u = urlparse(url)
    if u.hostname not in ("localhost", "127.0.0.1"):
        return url
    su = urlparse(server_url.rstrip("/"))
    return urlunparse((su.scheme, su.netloc, u.path, u.params, u.query, u.fragment))


DEFAULT_RESULTS = Path("results/bird_e2e.jsonl")
MAX_RESULT_ROWS = 500_000
STREAM_ROWS_PER_CHUNK = 8192
SCHEMA_MAX_CHARS = 2000
EVIDENCE_MAX_CHARS = 800
QUESTION_TRUNCATE = 400


@dataclass
class BirdE2EItem:
    question_id: str
    db_id: str
    question: str
    evidence: str
    gold_sql: str
    db_path: str


@dataclass
class BirdE2ERecord:
    question_id: str
    db_id: str
    question: str
    evidence: str
    sql_source: str = "gold"
    llm_backend: str = ""
    ollama_model: str = ""
    generated_sql: str = ""
    nl2sql_s: Optional[float] = None
    nl2sql_error: Optional[str] = None
    sql_exec_s: Optional[float] = None
    exec_ok: bool = False
    exec_error: Optional[str] = None
    used_gold_sql_for_transport: bool = False
    executed_sql: str = ""
    n_rows: int = 0
    n_cols: int = 0
    register_s: Optional[float] = None
    result_id: str = ""
    register_error: Optional[str] = None
    baseline_fetch_s: Optional[float] = None
    baseline_bytes: Optional[int] = None
    baseline_error: Optional[str] = None
    describe_s: Optional[float] = None
    recommended_format: str = ""
    enhanced_fetch_s: Optional[float] = None
    enhanced_bytes: Optional[int] = None
    enhanced_error: Optional[str] = None
    server_auto_call_s: Optional[float] = None
    server_auto_payload_s: Optional[float] = None
    server_auto_chosen_format: str = ""
    server_auto_bytes: Optional[int] = None
    server_auto_error: Optional[str] = None
    # Round-2 (F9): inline arm = bird_query_run_inline (no /materialized POST,
    # no parquet disk write on JSON path). Mirrors the server_auto_* shape.
    inline_call_s: Optional[float] = None
    inline_payload_s: Optional[float] = None
    inline_chosen_format: str = ""
    inline_bytes: Optional[int] = None
    inline_error: Optional[str] = None
    summary_s: Optional[float] = None
    summary_text: str = ""
    summary_error: Optional[str] = None
    arms_run: str = "both"
    strict_exec: bool = True
    frozen_sql_missing: bool = False


def load_bird_e2e_items(
    data_dir: Path,
    questions_file: str,
    max_queries: int,
) -> List[BirdE2EItem]:
    qpath = Path(questions_file)
    if not qpath.is_absolute():
        qpath = (data_dir / questions_file).resolve()
    if not qpath.is_file():
        log.error("Questions file not found: %s", qpath)
        return []
    with open(qpath, encoding="utf-8") as f:
        items = json.load(f)
    out: List[BirdE2EItem] = []
    for item in items:
        if len(out) >= max_queries:
            break
        db_id = item.get("db_id", "")
        gold = item.get("SQL", item.get("sql", ""))
        if not db_id or not gold:
            continue
        resolved = resolve_bird_sqlite_path(data_dir, db_id)
        if resolved is None:
            continue
        qid = str(item.get("question_id", len(out)))
        q = str(item.get("question", ""))
        ev = str(item.get("evidence", "") or "")
        out.append(
            BirdE2EItem(
                question_id=qid,
                db_id=db_id,
                question=q,
                evidence=ev,
                gold_sql=gold,
                db_path=str(resolved),
            )
        )
    return out


def _row_sql(obj: Dict[str, Any]) -> str:
    return str(obj.get("SQL") or obj.get("sql") or "").strip()


def load_frozen_sql_map(path: Path) -> Dict[Tuple[str, str], str]:
    """
    Load mapping (question_id, db_id) -> SQL string.

    Supports:
    - JSONL: one JSON object per line with question_id, db_id, and sql or SQL.
    - JSON array file: same objects in a top-level array.
    """
    text = path.read_text(encoding="utf-8").strip()
    rows: List[Dict[str, Any]]
    if not text:
        log.error("Frozen SQL file is empty: %s", path)
        return {}
    if text.startswith("["):
        rows = json.loads(text)
        if not isinstance(rows, list):
            log.error("Frozen SQL JSON must be an array: %s", path)
            return {}
    else:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    out: Dict[Tuple[str, str], str] = {}
    for obj in rows:
        if not isinstance(obj, dict):
            continue
        qid = str(obj.get("question_id", "")).strip()
        db = str(obj.get("db_id", "")).strip()
        sql = _row_sql(obj)
        if not qid or not db or not sql:
            continue
        out[(qid, db)] = sql
    log.info("Loaded %d frozen SQL rows from %s", len(out), path)
    return out


def schema_snippet_for_db(sqlite_path: Path, max_chars: int = SCHEMA_MAX_CHARS) -> str:
    desc = sqlite_path.parent / "database_description"
    if not desc.is_dir():
        return ""
    parts: List[str] = []
    for csv_path in sorted(desc.glob("*.csv"))[:6]:
        try:
            chunk = csv_path.read_text(encoding="utf-8", errors="replace")[:500]
            parts.append(f"--- {csv_path.name} ---\n{chunk}")
        except OSError:
            continue
    text = "\n".join(parts)
    return text[:max_chars] if text else ""


def extract_sql(raw: str) -> str:
    s = raw.strip()
    fence = re.search(r"```(?:sql)?\s*([\s\S]*?)```", s, re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()
    s = s.strip().strip("`").strip()
    if ";" in s:
        s = s.split(";")[0].strip()
    return s


def build_nl2sql_prompt(item: BirdE2EItem, schema: str) -> str:
    ev = (item.evidence or "")[:EVIDENCE_MAX_CHARS]
    q = (item.question or "")[:QUESTION_TRUNCATE]
    schema_block = schema or "(no schema excerpt on disk; use evidence and question only.)"
    return (
        "You are a SQLite expert. Output exactly one executable SELECT query. "
        "Use double quotes for identifiers when they are reserved or contain spaces. "
        "No markdown fences, no explanation, no semicolon after the query.\n\n"
        f"Database id: {item.db_id}\n"
        f"Schema excerpt (CSV fragments):\n{schema_block}\n\n"
        f"Evidence:\n{ev}\n\n"
        f"Question:\n{q}\n\n"
        "SQL:"
    )


async def nl2sql_ollama(prompt: str) -> tuple[str, float, Optional[str]]:
    t0 = time.perf_counter()
    try:
        raw = await chat([{"role": "user", "content": prompt}])
        elapsed = time.perf_counter() - t0
        sql = extract_sql(raw)
        if not sql:
            return "", elapsed, "empty_sql_after_extract"
        return sql, elapsed, None
    except Exception as exc:
        return "", time.perf_counter() - t0, str(exc)


async def baseline_fetch_json(
    session: Any,
    result_id: str,
    n_rows: int,
    n_cols: int,
) -> tuple[float, int]:
    t0 = time.perf_counter()
    structured = await call_large_json(session, n_rows, n_cols, result_id=result_id)
    raw = json.dumps(structured)
    elapsed = time.perf_counter() - t0
    return elapsed, len(raw.encode("utf-8"))


async def enhanced_fetch_recommended(
    session: Any,
    http: httpx.AsyncClient,
    result_id: str,
    n_rows: int,
    n_cols: int,
    target: OptimizationTarget,
) -> tuple[float, float, str, int]:
    """
    Returns (describe_s, fetch_s, recommended_format, bytes).
    """
    t_desc = time.perf_counter()
    hints = await call_describe_result_formats(
        session,
        n_rows,
        n_cols,
        rows_per_chunk=min(STREAM_ROWS_PER_CHUNK, max(1, n_rows)),
        result_id=result_id,
        optimization_target=target.value,
        prefer_streaming=False,
    )
    describe_s = time.perf_counter() - t_desc

    ctx = SelectionContext(n_rows=n_rows, n_cols=n_cols, target=target)
    if hints:
        recommended = select_format_with_hints(ctx, hints)
    else:
        recommended = select_format(ctx)

    t_fetch = time.perf_counter()
    if recommended == "json":
        structured = await call_large_json(session, n_rows, n_cols, result_id=result_id)
        raw = json.dumps(structured)
        nbytes = len(raw.encode("utf-8"))
    elif recommended == "parquet_blob":
        desc = await call_large_parquet_blob(session, n_rows, n_cols, result_id=result_id)
        data = await fetch_blob(http, desc["url"])
        nbytes = len(data)
    elif recommended == "arrow_ipc_blob":
        desc = await call_large_arrow_ipc_blob(session, n_rows, n_cols, result_id=result_id)
        data = await fetch_blob(http, desc["url"])
        nbytes = len(data)
    elif recommended == "arrow_ipc_stream":
        rpc = min(STREAM_ROWS_PER_CHUNK, max(1, n_rows))
        desc = await call_large_arrow_ipc_stream(
            session, n_rows, n_cols, rows_per_chunk=rpc, result_id=result_id,
        )
        url = desc["url"]
        nbytes = 0
        prefix = 8
        async for chunk in fetch_stream_chunks(http, url, length_prefix_bytes=prefix):
            nbytes += prefix + len(chunk)
    else:
        rpc = min(STREAM_ROWS_PER_CHUNK, max(1, n_rows))
        desc = await call_large_parquet_stream(
            session, n_rows, n_cols, rows_per_chunk=rpc, result_id=result_id,
        )
        url = desc["url"]
        nbytes = 0
        prefix = 8
        async for chunk in fetch_stream_chunks(http, url, length_prefix_bytes=prefix):
            nbytes += prefix + len(chunk)
    fetch_s = time.perf_counter() - t_fetch
    return describe_s, fetch_s, recommended, nbytes


async def inline_transport(
    session: Any,
    http: httpx.AsyncClient,
    db_id: str,
    sql: str,
    target: OptimizationTarget,
    server_url: str,
    rows_per_chunk: int,
) -> tuple[float, float, str, int]:
    """
    Round-2 (F9): one MCP round trip executing SQL + selecting format +
    returning payload via `bird_query_run_inline`. No /materialized POST.

    Returns (call_s, payload_s, chosen_format, nbytes).
    """
    t_call = time.perf_counter()
    auto = await call_bird_query_run_inline(
        session,
        db_id=db_id,
        sql=sql,
        optimization_target=target.value,
        rows_per_chunk=rows_per_chunk,
        prefer_streaming=False,
        use_mab=False,
    )
    call_s = time.perf_counter() - t_call

    t_pay = time.perf_counter()
    payload = auto.get("payload") if isinstance(auto, dict) else None
    decode = auto.get("decode") if isinstance(auto, dict) else None
    chosen = str(auto.get("chosen_format") or "")

    if isinstance(payload, dict) and payload.get("kind") == "json":
        raw = json.dumps(payload.get("records"))
        nbytes = len(raw.encode("utf-8"))
        return call_s, time.perf_counter() - t_pay, chosen, nbytes

    if not isinstance(decode, dict) or not isinstance(decode.get("url"), str):
        return call_s, time.perf_counter() - t_pay, chosen, 0

    url = _rewrite_server_url(decode["url"], server_url)
    transport = str(decode.get("transport") or "")
    prefix = 8
    if transport == "http_length_prefixed_stream":
        nbytes = 0
        async for chunk in fetch_stream_chunks(http, url, length_prefix_bytes=prefix):
            nbytes += prefix + len(chunk)
    else:
        data = await fetch_blob(http, url)
        nbytes = len(data)
    return call_s, time.perf_counter() - t_pay, chosen, nbytes


async def server_auto_transport(
    session: Any,
    http: httpx.AsyncClient,
    result_id: str,
    n_rows: int,
    n_cols: int,
    target: OptimizationTarget,
    server_url: str,
) -> tuple[float, float, str, int]:
    """
    One MCP round trip: large_result_auto (server picks format), then optional
    HTTP fetch for blob/stream descriptors. Inline JSON only json.dumps for byte count.

    Returns (call_s, payload_s, chosen_format, nbytes).
    """
    rpc = min(STREAM_ROWS_PER_CHUNK, max(1, n_rows))
    t_call = time.perf_counter()
    auto = await call_large_result_auto(
        session,
        n_rows,
        n_cols,
        rows_per_chunk=rpc,
        result_id=result_id,
        optimization_target=target.value,
        prefer_streaming=False,
        use_mab=False,
    )
    call_s = time.perf_counter() - t_call

    t_pay = time.perf_counter()
    payload = auto.get("payload") if isinstance(auto, dict) else None
    decode = auto.get("decode") if isinstance(auto, dict) else None
    chosen = str(auto.get("chosen_format") or "")

    if isinstance(payload, dict) and payload.get("kind") == "json":
        raw = json.dumps(payload.get("records"))
        nbytes = len(raw.encode("utf-8"))
        payload_s = time.perf_counter() - t_pay
        return call_s, payload_s, chosen, nbytes

    if isinstance(payload, dict) and payload.get("kind") == "text":
        text = payload.get("text") or ""
        nbytes = len(str(text).encode("utf-8"))
        payload_s = time.perf_counter() - t_pay
        return call_s, payload_s, chosen, nbytes

    if not isinstance(decode, dict) or not isinstance(decode.get("url"), str):
        payload_s = time.perf_counter() - t_pay
        return call_s, payload_s, chosen, 0

    url = _rewrite_server_url(decode["url"], server_url)
    transport = str(decode.get("transport") or "")
    prefix = 8
    if transport == "http_length_prefixed_stream":
        nbytes = 0
        async for chunk in fetch_stream_chunks(
            http, url, length_prefix_bytes=prefix,
        ):
            nbytes += prefix + len(chunk)
    else:
        data = await fetch_blob(http, url)
        nbytes = len(data)
    payload_s = time.perf_counter() - t_pay
    return call_s, payload_s, chosen, nbytes


async def optional_summary_llm(question: str, df: pd.DataFrame) -> tuple[float, str, Optional[str]]:
    sample = df.head(3)
    sample_str = sample.to_string(max_rows=3, max_cols=12)[:1500]
    prompt = (
        "In one or two short sentences, describe what the query result shows "
        "in relation to the user's question. Be factual.\n\n"
        f"Question: {question[:QUESTION_TRUNCATE]}\n"
        f"Result sample (first rows):\n{sample_str}"
    )
    t0 = time.perf_counter()
    try:
        text = (await chat([{"role": "user", "content": prompt}])).strip()
        return time.perf_counter() - t0, text[:4000], None
    except Exception as exc:
        return time.perf_counter() - t0, "", str(exc)


def append_jsonl(path: Path, rec: BirdE2ERecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")


async def run_one(
    session: Any,
    http: httpx.AsyncClient,
    item: BirdE2EItem,
    *,
    server_url: str,
    sql_source: Literal["gold", "jsonl", "ollama"],
    frozen_sql_map: Optional[Dict[Tuple[str, str], str]],
    strict_exec: bool,
    allow_gold_fallback: bool,
    arms: str,
    with_summary: bool,
    target: OptimizationTarget,
) -> BirdE2ERecord:
    use_llm = sql_source == "ollama" or with_summary
    rec = BirdE2ERecord(
        question_id=item.question_id,
        db_id=item.db_id,
        question=item.question[:800],
        evidence=(item.evidence or "")[:EVIDENCE_MAX_CHARS],
        sql_source=sql_source,
        llm_backend=get_llm_backend() if use_llm else "",
        ollama_model=os.environ.get("OLLAMA_MODEL", OLLAMA_MODEL) if use_llm else "",
        arms_run=arms,
        strict_exec=strict_exec,
    )

    sql_to_run = ""
    nl2sql_s: Optional[float] = None
    nl2sql_err: Optional[str] = None

    if sql_source == "gold":
        sql_to_run = bird_sql_for_sqlite(item.gold_sql)
        rec.generated_sql = item.gold_sql[:4000]
    elif sql_source == "jsonl":
        key = (str(item.question_id).strip(), str(item.db_id).strip())
        frozen = (frozen_sql_map or {}).get(key)
        if not frozen:
            rec.frozen_sql_missing = True
            rec.exec_error = f"no_frozen_sql_for_key={key!r}"
            return rec
        sql_to_run = bird_sql_for_sqlite(frozen)
        rec.generated_sql = frozen[:4000]
    else:
        schema = schema_snippet_for_db(Path(item.db_path))
        prompt = build_nl2sql_prompt(item, schema)
        gen_sql, nl2sql_s, nl2sql_err = await nl2sql_ollama(prompt)
        rec.generated_sql = gen_sql[:4000]
        rec.nl2sql_s = nl2sql_s
        rec.nl2sql_error = nl2sql_err
        if gen_sql and not nl2sql_err:
            sql_to_run = bird_sql_for_sqlite(gen_sql)

    exec_ok = False
    df: Optional[pd.DataFrame] = None
    sql_exec_s: Optional[float] = None

    if sql_to_run:
        try:
            t0 = time.perf_counter()
            df = execute_sql_sqlite(item.db_path, sql_to_run, max_rows=MAX_RESULT_ROWS)
            sql_exec_s = time.perf_counter() - t0
            exec_ok = True
            rec.executed_sql = sql_to_run[:4000]
        except Exception as exc:
            rec.exec_error = str(exc)
            sql_exec_s = None

    if not exec_ok and sql_source == "ollama" and allow_gold_fallback and item.gold_sql:
        try:
            gold = bird_sql_for_sqlite(item.gold_sql)
            t0 = time.perf_counter()
            df = execute_sql_sqlite(item.db_path, gold, max_rows=MAX_RESULT_ROWS)
            sql_exec_s = time.perf_counter() - t0
            exec_ok = True
            rec.used_gold_sql_for_transport = True
            rec.executed_sql = gold[:4000]
        except Exception as exc:
            rec.exec_error = (rec.exec_error or "") + f" | gold_fallback: {exc}"

    if not exec_ok:
        rec.sql_exec_s = sql_exec_s
        rec.exec_ok = False
        return rec

    if df is None or df.empty:
        rec.exec_ok = False
        rec.sql_exec_s = sql_exec_s
        rec.exec_error = rec.exec_error or "empty_result"
        return rec

    rec.exec_ok = True
    rec.sql_exec_s = sql_exec_s
    n_rows, n_cols = df.shape
    rec.n_rows = int(n_rows)
    rec.n_cols = int(n_cols)

    # Inline arm bypasses the /materialized POST entirely — the server-side
    # `bird_query_run_inline` tool does the SQL execution itself. We still
    # need a result_id only when the user explicitly opted into a non-inline
    # arm.
    needs_register = arms in ("both", "baseline", "enhanced", "server")
    if needs_register:
        try:
            t0 = time.perf_counter()
            pq_bytes = df_to_parquet_bytes(df)
            reg = await register_materialized(http, pq_bytes, base_url=server_url)
            rec.register_s = time.perf_counter() - t0
            rec.result_id = reg.get("result_id", "")
        except Exception as exc:
            rec.register_error = str(exc)
            return rec

    result_id = rec.result_id
    if arms in ("both", "baseline"):
        try:
            bf, bb = await baseline_fetch_json(session, result_id, n_rows, n_cols)
            rec.baseline_fetch_s = bf
            rec.baseline_bytes = bb
        except Exception as exc:
            rec.baseline_error = str(exc)

    if arms in ("both", "enhanced"):
        try:
            ds, fs, fmt, nb = await enhanced_fetch_recommended(
                session, http, result_id, n_rows, n_cols, target,
            )
            rec.describe_s = ds
            rec.enhanced_fetch_s = fs
            rec.recommended_format = fmt
            rec.enhanced_bytes = nb
        except Exception as exc:
            rec.enhanced_error = str(exc)

    if arms == "server":
        try:
            cs, ps, ch_fmt, nb = await server_auto_transport(
                session,
                http,
                result_id,
                n_rows,
                n_cols,
                target,
                server_url,
            )
            rec.server_auto_call_s = cs
            rec.server_auto_payload_s = ps
            rec.server_auto_chosen_format = ch_fmt
            rec.server_auto_bytes = nb
        except Exception as exc:
            rec.server_auto_error = str(exc)

    if arms == "inline":
        try:
            rpc = min(STREAM_ROWS_PER_CHUNK, max(1, int(n_rows)))
            cs, ps, ch_fmt, nb = await inline_transport(
                session,
                http,
                item.db_id,
                sql_to_run,
                target,
                server_url,
                rows_per_chunk=rpc,
            )
            rec.inline_call_s = cs
            rec.inline_payload_s = ps
            rec.inline_chosen_format = ch_fmt
            rec.inline_bytes = nb
        except Exception as exc:
            rec.inline_error = str(exc)

    if with_summary and df is not None and not df.empty:
        ss, st, se = await optional_summary_llm(item.question, df)
        rec.summary_s = ss
        rec.summary_text = st
        rec.summary_error = se

    return rec


async def run_bench(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).resolve()
    target = get_default_target()

    frozen_map: Optional[Dict[Tuple[str, str], str]] = None
    if args.sql_source == "jsonl":
        if not args.frozen_sql:
            log.error("--frozen-sql is required when --sql-source jsonl")
            return
        fp = Path(args.frozen_sql).resolve()
        if not fp.is_file():
            log.error("Frozen SQL file not found: %s", fp)
            return
        frozen_map = load_frozen_sql_map(fp)
        if not frozen_map:
            log.error("No valid rows in frozen SQL file")
            return

    items = load_bird_e2e_items(data_dir, args.bird_questions, args.max_queries)
    log.info("Loaded %d BIRD items (sql_source=%s)", len(items), args.sql_source)
    if not items:
        return

    out_path = Path(args.results)
    if args.overwrite and out_path.exists():
        out_path.unlink()

    meta = {
        "type": "bird_e2e_run_header",
        "sql_source": args.sql_source,
        "frozen_sql": str(Path(args.frozen_sql).resolve()) if args.frozen_sql else None,
        "format_select_target": target.value,
        "arms": args.arms,
        "strict_exec": not args.allow_gold_fallback,
        "allow_gold_fallback": args.allow_gold_fallback,
        "with_summary": args.with_summary,
        "mcp_url": args.mcp_url,
        "server_url": args.server_url,
        "bird_questions": args.bird_questions,
        "max_queries": args.max_queries,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    async with connect(base_url=args.mcp_url) as (session, http):
        for i, item in enumerate(items):
            log.info("[%d/%d] qid=%s db=%s", i + 1, len(items), item.question_id, item.db_id)
            rec = await run_one(
                session,
                http,
                item,
                server_url=args.server_url,
                sql_source=args.sql_source,
                frozen_sql_map=frozen_map,
                strict_exec=not args.allow_gold_fallback,
                allow_gold_fallback=args.allow_gold_fallback,
                arms=args.arms,
                with_summary=args.with_summary,
                target=target,
            )
            append_jsonl(out_path, rec)
            log.info(
                "  exec_ok=%s rows=%s baseline=%s enhanced=%s server_auto=%s inline=%s",
                rec.exec_ok,
                rec.n_rows,
                rec.baseline_fetch_s is not None,
                rec.enhanced_fetch_s is not None,
                rec.server_auto_call_s is not None,
                rec.inline_call_s is not None,
            )


def main() -> None:
    p = argparse.ArgumentParser(
        description="BIRD transport: fixed SQL (gold or frozen JSONL) or optional Ollama NL2SQL",
    )
    p.add_argument("--data-dir", type=Path, default=Path("data/datasets/bird/dev"))
    p.add_argument("--bird-questions", default="mini_dev_sqlite.json")
    p.add_argument("--max-queries", type=int, default=50)
    p.add_argument("--mcp-url", default=DEFAULT_MCP_URL)
    p.add_argument("--server-url", default="http://localhost:8000")
    p.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--sql-source",
        choices=["gold", "jsonl", "ollama"],
        default="gold",
        help="gold=BIRD gold SQL (core eval); jsonl=frozen SQL file; ollama=NL2SQL (secondary)",
    )
    p.add_argument(
        "--frozen-sql",
        type=Path,
        default=None,
        help="JSONL or JSON array with question_id, db_id, sql/SQL (required for --sql-source jsonl)",
    )
    p.add_argument(
        "--arms",
        choices=["both", "baseline", "enhanced", "server", "inline"],
        default="both",
        help=(
            "Which transport arm(s) to run per query: baseline=large_json only; "
            "enhanced=describe+fetch; server=large_result_auto (server-side selection); "
            "inline=bird_query_run_inline (round-2 fused path: SQL+select+payload in 1 RTT); "
            "both=baseline+enhanced"
        ),
    )
    p.add_argument(
        "--allow-gold-fallback",
        action="store_true",
        help="(Ollama mode only) If generated SQL fails, execute gold SQL for transport",
    )
    p.add_argument(
        "--with-summary",
        action="store_true",
        help="After transport, call LLM once to summarize a fixed row sample",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Opt-in client-side pyinstrument profile: wraps the whole bench loop
    # (NL2SQL / SQL exec / MCP roundtrips). Outputs land under
    # results/profiling/client/. Enabled only when PYINSTRUMENT_PROFILE=1.
    if os.environ.get("PYINSTRUMENT_PROFILE", "").strip() == "1":
        from pyinstrument import Profiler
        from pyinstrument.renderers import SpeedscopeRenderer

        out_dir = Path(
            os.environ.get(
                "PYINSTRUMENT_OUT_DIR", "results/profiling/client"
            )
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = os.environ.get(
            "PYINSTRUMENT_TAG",
            f"bird_e2e_{args.sql_source}_{args.arms}_{args.max_queries}",
        )

        profiler = Profiler(async_mode="enabled", interval=0.0005)
        profiler.start()
        try:
            asyncio.run(run_bench(args))
        finally:
            profiler.stop()
            html_path = out_dir / f"{tag}.html"
            speedscope_path = out_dir / f"{tag}.speedscope.json"
            text_path = out_dir / f"{tag}.txt"
            html_path.write_text(profiler.output_html(), encoding="utf-8")
            speedscope_path.write_text(
                profiler.output(renderer=SpeedscopeRenderer()),
                encoding="utf-8",
            )
            text_path.write_text(
                profiler.output_text(unicode=True, color=False, show_all=False),
                encoding="utf-8",
            )
            log.info(
                "pyinstrument: wrote %s, %s, %s",
                html_path, speedscope_path, text_path,
            )
        return

    asyncio.run(run_bench(args))


if __name__ == "__main__":
    main()
