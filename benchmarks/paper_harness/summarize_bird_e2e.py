"""
Summarize results/bird_e2e.jsonl into markdown + JSON.

Usage:

    python summarize_bird_e2e.py
    python summarize_bird_e2e.py --input results/bird_e2e.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_INPUT = Path("results/bird_e2e.jsonl")
DEFAULT_MD = Path("results/bird_e2e_summary.md")
DEFAULT_JSON = Path("results/bird_e2e_summary.json")


def _median(vals: Sequence[Optional[float]]) -> Optional[float]:
    xs = [float(x) for x in vals if x is not None]
    if not xs:
        return None
    return float(statistics.median(xs))


def _p95(vals: Sequence[Optional[float]]) -> Optional[float]:
    xs = sorted(float(x) for x in vals if x is not None)
    if not xs:
        return None
    i = max(0, min(len(xs) - 1, int(round(0.95 * (len(xs) - 1)))))
    return xs[i]


def load_records(path: Path) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if not path.exists():
        return [], None
    rows: List[Dict[str, Any]] = []
    header: Optional[Dict[str, Any]] = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "bird_e2e_run_header":
                header = obj
                continue
            rows.append(obj)
    return rows, header


def summarize(rows: List[Dict[str, Any]], header: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    # Inline arm doesn't allocate a result_id (no /materialized POST), so
    # widen the inclusion criterion when an inline_call_s is present.
    ok_transport = [
        r
        for r in rows
        if r.get("exec_ok")
        and (r.get("result_id") or r.get("inline_call_s") is not None)
        and (
            r.get("baseline_fetch_s") is not None
            or r.get("enhanced_fetch_s") is not None
            or r.get("server_auto_call_s") is not None
            or r.get("inline_call_s") is not None
        )
    ]
    exec_ok_count = sum(1 for r in rows if r.get("exec_ok"))
    reg_ok = sum(1 for r in rows if r.get("result_id"))
    gold_transport = sum(1 for r in rows if r.get("used_gold_sql_for_transport"))

    def nums(key: str, rs: List[Dict[str, Any]] = rows) -> List[Optional[float]]:
        return [r.get(key) for r in rs]

    baseline_ok = [r for r in ok_transport if r.get("baseline_fetch_s") is not None]
    enhanced_ok = [r for r in ok_transport if r.get("enhanced_fetch_s") is not None]
    server_ok = [r for r in ok_transport if r.get("server_auto_call_s") is not None]
    inline_ok = [r for r in ok_transport if r.get("inline_call_s") is not None]

    ratio_fetch: List[float] = []
    ratio_bytes: List[float] = []
    enhanced_smaller = 0
    baseline_smaller = 0
    bytes_equal = 0
    total_bb = 0
    total_eb = 0
    for r in ok_transport:
        b, e = r.get("baseline_fetch_s"), r.get("enhanced_fetch_s")
        if b is not None and e is not None and e > 0:
            ratio_fetch.append(float(b) / float(e))
        bb, eb = r.get("baseline_bytes"), r.get("enhanced_bytes")
        if bb is not None and eb is not None:
            total_bb += int(bb)
            total_eb += int(eb)
            if eb > 0:
                ratio_bytes.append(float(bb) / float(eb))
            if bb == eb:
                bytes_equal += 1
            elif eb < bb:
                enhanced_smaller += 1
            elif bb < eb:
                baseline_smaller += 1

    fmt_counts: Dict[str, int] = {}
    for r in enhanced_ok:
        f = r.get("recommended_format") or "unknown"
        fmt_counts[f] = fmt_counts.get(f, 0) + 1

    server_fmt_counts: Dict[str, int] = {}
    for r in server_ok:
        f = r.get("server_auto_chosen_format") or "unknown"
        server_fmt_counts[f] = server_fmt_counts.get(f, 0) + 1

    inline_fmt_counts: Dict[str, int] = {}
    for r in inline_ok:
        f = r.get("inline_chosen_format") or "unknown"
        inline_fmt_counts[f] = inline_fmt_counts.get(f, 0) + 1

    summaries = [
        (r.get("summary_text") or "", r.get("summary_error"))
        for r in rows
        if r.get("summary_s") is not None or r.get("summary_text") or r.get("summary_error")
    ]

    example_summaries: List[Dict[str, Any]] = []
    for r in rows[:15]:
        if r.get("summary_text") or r.get("summary_error"):
            example_summaries.append(
                {
                    "question_id": r.get("question_id"),
                    "db_id": r.get("db_id"),
                    "summary_text": (r.get("summary_text") or "")[:500],
                    "summary_error": r.get("summary_error"),
                }
            )

    nl2sql_errs = sum(1 for r in rows if r.get("nl2sql_error"))
    exec_errs = sum(1 for r in rows if r.get("exec_error") and not r.get("exec_ok"))
    frozen_missing = sum(1 for r in rows if r.get("frozen_sql_missing"))
    sql_source_counts: Dict[str, int] = {}
    for r in rows:
        src = r.get("sql_source") or "unknown"
        sql_source_counts[src] = sql_source_counts.get(src, 0) + 1

    return {
        "run_header": header,
        "total_query_records": len(rows),
        "exec_ok": exec_ok_count,
        "registration_ok": reg_ok,
        "transport_measured": len(ok_transport),
        "used_gold_sql_for_transport": gold_transport,
        "nl2sql_errors": nl2sql_errs,
        "exec_failures_no_row": exec_errs,
        "frozen_sql_missing": frozen_missing,
        "sql_source_counts": sql_source_counts,
        "recommended_format_counts": fmt_counts,
        "server_auto_format_counts": server_fmt_counts,
        "inline_format_counts": inline_fmt_counts,
        "median_nl2sql_s": _median(nums("nl2sql_s")),
        "p95_nl2sql_s": _p95(nums("nl2sql_s")),
        "median_baseline_fetch_s": _median([r.get("baseline_fetch_s") for r in baseline_ok]),
        "p95_baseline_fetch_s": _p95([r.get("baseline_fetch_s") for r in baseline_ok]),
        "median_describe_s": _median([r.get("describe_s") for r in enhanced_ok]),
        "p95_describe_s": _p95([r.get("describe_s") for r in enhanced_ok]),
        "median_enhanced_fetch_s": _median([r.get("enhanced_fetch_s") for r in enhanced_ok]),
        "p95_enhanced_fetch_s": _p95([r.get("enhanced_fetch_s") for r in enhanced_ok]),
        "median_enhanced_total_s": _median(
            [
                (float(r.get("describe_s") or 0) + float(r.get("enhanced_fetch_s") or 0))
                for r in enhanced_ok
                if r.get("describe_s") is not None and r.get("enhanced_fetch_s") is not None
            ]
        ),
        "p95_enhanced_total_s": _p95(
            [
                (float(r.get("describe_s") or 0) + float(r.get("enhanced_fetch_s") or 0))
                for r in enhanced_ok
                if r.get("describe_s") is not None and r.get("enhanced_fetch_s") is not None
            ]
        ),
        "median_server_auto_call_s": _median([r.get("server_auto_call_s") for r in server_ok]),
        "p95_server_auto_call_s": _p95([r.get("server_auto_call_s") for r in server_ok]),
        "median_server_auto_payload_s": _median(
            [r.get("server_auto_payload_s") for r in server_ok]
        ),
        "p95_server_auto_payload_s": _p95([r.get("server_auto_payload_s") for r in server_ok]),
        "median_server_auto_total_s": _median(
            [
                float(r.get("server_auto_call_s") or 0)
                + float(r.get("server_auto_payload_s") or 0)
                for r in server_ok
                if r.get("server_auto_call_s") is not None
                and r.get("server_auto_payload_s") is not None
            ]
        ),
        "p95_server_auto_total_s": _p95(
            [
                float(r.get("server_auto_call_s") or 0)
                + float(r.get("server_auto_payload_s") or 0)
                for r in server_ok
                if r.get("server_auto_call_s") is not None
                and r.get("server_auto_payload_s") is not None
            ]
        ),
        "median_server_auto_bytes": _median([r.get("server_auto_bytes") for r in server_ok]),
        "median_inline_call_s": _median([r.get("inline_call_s") for r in inline_ok]),
        "p95_inline_call_s": _p95([r.get("inline_call_s") for r in inline_ok]),
        "median_inline_payload_s": _median([r.get("inline_payload_s") for r in inline_ok]),
        "p95_inline_payload_s": _p95([r.get("inline_payload_s") for r in inline_ok]),
        "median_inline_total_s": _median(
            [
                float(r.get("inline_call_s") or 0) + float(r.get("inline_payload_s") or 0)
                for r in inline_ok
                if r.get("inline_call_s") is not None
                and r.get("inline_payload_s") is not None
            ]
        ),
        "p95_inline_total_s": _p95(
            [
                float(r.get("inline_call_s") or 0) + float(r.get("inline_payload_s") or 0)
                for r in inline_ok
                if r.get("inline_call_s") is not None
                and r.get("inline_payload_s") is not None
            ]
        ),
        "median_inline_bytes": _median([r.get("inline_bytes") for r in inline_ok]),
        "median_baseline_bytes": _median([r.get("baseline_bytes") for r in baseline_ok]),
        "median_enhanced_bytes": _median([r.get("enhanced_bytes") for r in enhanced_ok]),
        "p95_baseline_bytes": _p95([float(r.get("baseline_bytes")) for r in baseline_ok if r.get("baseline_bytes") is not None]),
        "p95_enhanced_bytes": _p95([float(r.get("enhanced_bytes")) for r in enhanced_ok if r.get("enhanced_bytes") is not None]),
        "payload_queries_where_bytes_equal": bytes_equal,
        "payload_queries_where_enhanced_smaller": enhanced_smaller,
        "payload_queries_where_baseline_smaller": baseline_smaller,
        "payload_sum_baseline_bytes": total_bb,
        "payload_sum_enhanced_bytes": total_eb,
        "median_ratio_baseline_fetch_over_enhanced_fetch": _median(ratio_fetch)
        if ratio_fetch
        else None,
        "median_ratio_baseline_bytes_over_enhanced_bytes": _median(ratio_bytes)
        if ratio_bytes
        else None,
        "median_summary_s": _median(nums("summary_s")),
        "p95_summary_s": _p95(nums("summary_s")),
        "summary_examples": example_summaries,
        "environment": {
            "ollama_model": os.environ.get("OLLAMA_MODEL", "(not set in summarize env)"),
        },
    }


CACHING_SECTION = """
## Caching and measurement fairness

- **Materialized hints + payload cache (round-2):** for `POST /materialized` and `bird_query_materialize`/`bird_query_run_inline` results, both size hints AND the materialized DataFrame, Arrow table, JSON records (small payloads), and encoded Parquet/Arrow IPC bytes are pre-computed at registration (`ResultConfig.cached_*` fields in `server_app.py`), so `describe_result_formats`, `large_result_auto`, `large_json`, and the parquet/IPC HTTP blob endpoints all avoid re-reading Parquet and re-encoding on the hot path. The cache is bounded by `RESULT_CACHE_MAX_BYTES` (default 64 MB). Synthetic (no `result_id`) LRU caches still apply only when `describe_result_formats` is called without a materialized `result_id`.
- **JSON-Schema validation backend (round-2):** the python-sdk client/server now picks the fastest available validator at import: `MCP_VALIDATOR_BACKEND={auto|jsonschema-rs|fastjsonschema|jsonschema|skip}` (or `MCP_SKIP_VALIDATE=1` for the no-op fast path). Default `auto` prefers `jsonschema-rs` then `fastjsonschema` then `jsonschema`. See `mcp.shared._validation`.
- **HTTP keep-alive** (httpx) may shave a small amount off repeated localhost RPCs; arms share the same session per query, so relative comparison on the **same** `result_id` remains meaningful.
- **SQLite / OS page cache** can speed up repeated access to the same DB file across queries; absolute SQL times are “warm-ish” after the first touch on a given database.
- **Ollama** may reuse KV cache for similar prompt prefixes; NL2SQL prompts share a template, so later queries might be slightly faster. Report the model name and note whether the daemon was restarted between experiments if you need stricter cold behavior.
"""


def write_markdown(data: Dict[str, Any], path: Path) -> None:
    h = data.get("run_header") or {}
    lines = [
        "# BIRD end-to-end benchmark summary",
        "",
        "## Run configuration",
        "",
        f"- **Query records:** {data.get('total_query_records', 0)}",
        f"- **Exec OK:** {data.get('exec_ok', 0)}",
        f"- **Registration OK:** {data.get('registration_ok', 0)}",
        f"- **Transport measured (both arms attempted):** {data.get('transport_measured', 0)}",
        f"- **Gold SQL used for transport (fallback):** {data.get('used_gold_sql_for_transport', 0)}",
        f"- **NL2SQL errors:** {data.get('nl2sql_errors', 0)}",
        f"- **Strict exec failures (no row):** {data.get('exec_failures_no_row', 0)}",
        f"- **Frozen SQL missing (jsonl key):** {data.get('frozen_sql_missing', 0)}",
    ]
    ssc = data.get("sql_source_counts") or {}
    if ssc:
        lines.append(f"- **sql_source counts:** {ssc}")
    if h:
        lines.extend(
            [
                f"- **SQL source:** {h.get('sql_source', 'n/a')}",
                f"- **Frozen SQL file:** {h.get('frozen_sql', 'n/a')}",
                f"- **Arms:** {h.get('arms', 'n/a')}",
                f"- **Allow gold fallback:** {h.get('allow_gold_fallback', 'n/a')}",
                f"- **With summary LLM:** {h.get('with_summary', 'n/a')}",
                f"- **Format select target:** {h.get('format_select_target', 'n/a')}",
                f"- **MCP URL:** {h.get('mcp_url', 'n/a')}",
            ]
        )
    lines.extend(
        [
            "",
            "## Latency (seconds, median / p95 where noted)",
            "",
            f"| Stage | Median | p95 |",
            f"|---|---:|---:|",
            f"| NL2SQL | {_fmt(data.get('median_nl2sql_s'))} | {_fmt(data.get('p95_nl2sql_s'))} |",
            f"| Baseline fetch (JSON only) | {_fmt(data.get('median_baseline_fetch_s'))} | {_fmt(data.get('p95_baseline_fetch_s'))} |",
            f"| Describe (enhanced) | {_fmt(data.get('median_describe_s'))} | {_fmt(data.get('p95_describe_s'))} |",
            f"| Enhanced fetch (chosen format) | {_fmt(data.get('median_enhanced_fetch_s'))} | {_fmt(data.get('p95_enhanced_fetch_s'))} |",
            f"| Enhanced describe + fetch | {_fmt(data.get('median_enhanced_total_s'))} | {_fmt(data.get('p95_enhanced_total_s'))} |",
            f"| `large_result_auto` call (server arm) | {_fmt(data.get('median_server_auto_call_s'))} | {_fmt(data.get('p95_server_auto_call_s'))} |",
            f"| Server auto payload fetch / size | {_fmt(data.get('median_server_auto_payload_s'))} | {_fmt(data.get('p95_server_auto_payload_s'))} |",
            f"| Server auto call + payload | {_fmt(data.get('median_server_auto_total_s'))} | {_fmt(data.get('p95_server_auto_total_s'))} |",
            f"| `bird_query_run_inline` call (inline arm) | {_fmt(data.get('median_inline_call_s'))} | {_fmt(data.get('p95_inline_call_s'))} |",
            f"| Inline payload fetch / size | {_fmt(data.get('median_inline_payload_s'))} | {_fmt(data.get('p95_inline_payload_s'))} |",
            f"| Inline call + payload | {_fmt(data.get('median_inline_total_s'))} | {_fmt(data.get('p95_inline_total_s'))} |",
            f"| Optional summary LLM | {_fmt(data.get('median_summary_s'))} | {_fmt(data.get('p95_summary_s'))} |",
            "",
            "## Payload sizes (bytes)",
            "",
            f"| | Median | p95 |",
            f"|---|---:|---:|",
            f"| Baseline JSON | {_fmt(data.get('median_baseline_bytes'))} | {_fmt(data.get('p95_baseline_bytes'))} |",
            f"| Enhanced (chosen format) | {_fmt(data.get('median_enhanced_bytes'))} | {_fmt(data.get('p95_enhanced_bytes'))} |",
            "",
            f"- **Queries with equal baseline vs enhanced bytes:** {data.get('payload_queries_where_bytes_equal', '—')}",
            f"- **Queries where enhanced payload is smaller than baseline:** {data.get('payload_queries_where_enhanced_smaller', '—')}",
            f"- **Queries where baseline is smaller than enhanced:** {data.get('payload_queries_where_baseline_smaller', '—')}",
            f"- **Sum of baseline bytes (all successful transport rows):** {data.get('payload_sum_baseline_bytes', '—')}",
            f"- **Sum of enhanced bytes (same):** {data.get('payload_sum_enhanced_bytes', '—')}",
            "",
            "## Ratios (median)",
            "",
            f"| baseline_fetch_s / enhanced_fetch_s | {_fmt(data.get('median_ratio_baseline_fetch_over_enhanced_fetch'))} |",
            f"| baseline_bytes / enhanced_bytes | {_fmt(data.get('median_ratio_baseline_bytes_over_enhanced_bytes'))} |",
            "",
            "## Recommended format (enhanced arm)",
            "",
        ]
    )
    for fmt, n in sorted((data.get("recommended_format_counts") or {}).items(), key=lambda x: -x[1]):
        lines.append(f"- `{fmt}`: {n}")
    sfc = data.get("server_auto_format_counts") or {}
    if sfc:
        lines.extend(
            [
                "",
                "## Chosen format (server arm, `large_result_auto`)",
                "",
            ]
        )
        for fmt, n in sorted(sfc.items(), key=lambda x: -x[1]):
            lines.append(f"- `{fmt}`: {n}")
        if data.get("median_server_auto_bytes") is not None:
            lines.append("")
            lines.append(
                f"- **Median payload bytes (after call):** {_fmt(data.get('median_server_auto_bytes'))}"
            )

    ifc = data.get("inline_format_counts") or {}
    if ifc:
        lines.extend(
            [
                "",
                "## Chosen format (inline arm, `bird_query_run_inline`)",
                "",
            ]
        )
        for fmt, n in sorted(ifc.items(), key=lambda x: -x[1]):
            lines.append(f"- `{fmt}`: {n}")
        if data.get("median_inline_bytes") is not None:
            lines.append("")
            lines.append(
                f"- **Median payload bytes (after call):** {_fmt(data.get('median_inline_bytes'))}"
            )
    lines.extend(
        [
            "",
            "## Example LLM summaries (first records with summary data)",
            "",
        ]
    )
    for ex in data.get("summary_examples") or []:
        qid = ex.get("question_id", "")
        db = ex.get("db_id", "")
        txt = ex.get("summary_text") or ""
        err = ex.get("summary_error")
        lines.append(f"### qid={qid} db={db}")
        if err:
            lines.append(f"- **Error:** {err}")
        if txt:
            lines.append("")
            lines.append(f"> {txt.replace(chr(10), ' ')}")
        lines.append("")

    lines.append(CACHING_SECTION.strip())
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.6f}".rstrip("0").rstrip(".")
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize bird_e2e.jsonl")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--md", type=Path, default=DEFAULT_MD)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    args = ap.parse_args()

    rows, header = load_records(args.input)
    data = summarize(rows, header)
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    write_markdown(data, args.md)
    print(f"Wrote {args.md} and {args.json_out} ({len(rows)} query records)")


if __name__ == "__main__":
    main()
