"""
AdaEdge-inspired format selection: choose JSON vs Parquet vs Arrow IPC (blob/stream)
using optimization target and data features (not just cell count).

See format_selection_inspired_by_adaedge.md and IMPLEMENTATION_PLAN_format_selection.md.

Environment (optional ACE-style latency for *min_latency* and TTFR stream pick):

- FORMAT_SELECT_TARGET — min_bytes | min_latency | min_time_to_first_rows (default min_latency).
- FORMAT_LATENCY_NETWORK_MBPS — if > 0, min_latency uses transfer_time(bytes) + decode_proxy(bytes)
  per format instead of raw min-bytes among blobs. 0 = keep min-bytes behavior (default).
- FORMAT_LATENCY_DECODE_NS_PER_BYTE_JSON, _PARQUET_BLOB, _ARROW_IPC_BLOB — override ns/byte decode proxy.
- FORMAT_LATENCY_CALIBRATION_JSON — path to JSON:
  {"decode_ns_per_byte": {"json": ..., "parquet_blob": ..., "arrow_ipc_blob": ...}}
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

FormatName = Literal[
    "json",
    "parquet_blob",
    "parquet_stream",
    "arrow_ipc_blob",
    "arrow_ipc_stream",
    # Unstructured payload arms (text/blob)
    "text_inline",
    "raw_blob",
    "gzip_blob",
]

# Hints from describe_result_formats: json_bytes, parquet_bytes, optional stream first chunks, IPC
HintsDict = dict[str, Any]


class OptimizationTarget(str, Enum):
    """What we optimize for when choosing format (AdaEdge: workload/target)."""

    MIN_BYTES = "min_bytes"  # Minimize payload size (bandwidth-limited)
    MIN_LATENCY = "min_latency"  # Minimize end-to-end time (default)
    MIN_TIME_TO_FIRST_ROWS = "min_time_to_first_rows"  # Streaming / agentic speculation


@dataclass
class SelectionContext:
    """Input to format selection: data shape, target, optional constraints."""

    n_rows: int
    n_cols: int
    target: OptimizationTarget = OptimizationTarget.MIN_LATENCY
    prefer_streaming: bool = False  # Hint: client wants early rows / early termination


# Tunable thresholds (can be replaced by lookup table or MAB later)
CELLS_SMALL = 50_000  # Below this, JSON is often acceptable
CELLS_MEDIUM = 300_000
CELLS_LARGE = 2_000_000
ROWS_STREAM_FAVOR = 50_000  # Above this, stream is better for time-to-first-rows

# Fix 4: when JSON bytes are at or below this threshold, JSON is the obvious
# winner under MIN_LATENCY / MIN_BYTES; the selector returns "json"
# without comparing parquet/IPC. Override via env.
JSON_OBVIOUS_WINNER_BYTES_DEFAULT = 4096


def _json_obvious_winner_bytes() -> int:
    raw = os.environ.get("FORMAT_HINTS_JSON_OBVIOUS_WINNER_BYTES", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return JSON_OBVIOUS_WINNER_BYTES_DEFAULT


def recommend_format_legacy(n_rows: int, n_cols: int) -> FormatName:
    """
    Legacy cell-count-only heuristic (kept for fallback and comparison).
    """
    total_cells = n_rows * n_cols
    if total_cells <= CELLS_SMALL:
        return "json"
    elif total_cells <= CELLS_MEDIUM:
        return "parquet_blob"
    elif total_cells <= CELLS_LARGE:
        return "parquet_blob"
    else:
        return "parquet_stream"


def select_format(context: SelectionContext) -> FormatName:
    """
    Choose format using optimization target and (n_rows, n_cols).
    Multi-factor rules; no IPC (hints required to compare arrow sizes).
    """
    n_rows = context.n_rows
    n_cols = context.n_cols
    target = context.target
    cells = n_rows * n_cols

    if target == OptimizationTarget.MIN_TIME_TO_FIRST_ROWS:
        if context.prefer_streaming or n_rows >= ROWS_STREAM_FAVOR:
            if cells > CELLS_SMALL:
                return "parquet_stream"
        if cells <= CELLS_SMALL:
            return "json"
        return "parquet_blob"

    if target == OptimizationTarget.MIN_BYTES:
        if cells <= CELLS_SMALL:
            return "json"
        if cells > CELLS_LARGE:
            return "parquet_stream"
        return "parquet_blob"

    if cells <= CELLS_SMALL:
        return "json"
    if cells <= CELLS_MEDIUM:
        return "parquet_blob"
    if cells <= CELLS_LARGE:
        return "parquet_blob"
    return "parquet_stream"


def _min_blob_format(
    json_bytes: int,
    parquet_bytes: int,
    arrow_ipc_bytes: Optional[int],
) -> FormatName:
    """Pick json, parquet_blob, or arrow_ipc_blob with smallest hinted size (tie: json < parquet < ipc)."""
    candidates: list[tuple[int, int, FormatName]] = [
        (json_bytes, 0, "json"),
        (parquet_bytes, 1, "parquet_blob"),
    ]
    if arrow_ipc_bytes is not None:
        candidates.append((arrow_ipc_bytes, 2, "arrow_ipc_blob"))
    return min(candidates, key=lambda t: (t[0], t[1]))[2]


def _get_network_mbps() -> float:
    """
    Effective link throughput for transfer-time term (ACE-style).
    0 = disabled; min_latency falls back to min-bytes among blob formats.
    Env: FORMAT_LATENCY_NETWORK_MBPS
    """
    raw = os.environ.get("FORMAT_LATENCY_NETWORK_MBPS", "0").strip()
    try:
        v = float(raw)
    except ValueError:
        return 0.0
    return max(0.0, v)


def _load_calibration_decode_ns() -> dict[str, float]:
    """
    Optional JSON from FORMAT_LATENCY_CALIBRATION_JSON, shape:
    {"decode_ns_per_byte": {"json": 1.0, "parquet_blob": 4.0, "arrow_ipc_blob": 0.5}}
    """
    path = os.environ.get("FORMAT_LATENCY_CALIBRATION_JSON", "").strip()
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    raw = data.get("decode_ns_per_byte")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _decode_ns_per_byte(fmt: Literal["json", "parquet_blob", "arrow_ipc_blob"]) -> float:
    """Decode-time proxy (ns/byte); calibration JSON overrides per-key env FORMAT_LATENCY_DECODE_NS_PER_BYTE_<FMT>."""
    cal = _load_calibration_decode_ns()
    if fmt in cal:
        return cal[fmt]
    defaults = {"json": 1.0, "parquet_blob": 4.0, "arrow_ipc_blob": 0.5}
    env_key = f"FORMAT_LATENCY_DECODE_NS_PER_BYTE_{fmt.upper()}"
    raw = os.environ.get(env_key, "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return defaults[fmt]


def _estimated_e2e_latency_seconds(blob_bytes: int, fmt: Literal["json", "parquet_blob", "arrow_ipc_blob"]) -> float:
    """transfer(bytes/BW) + decode_proxy(ns/B * bytes); BW from FORMAT_LATENCY_NETWORK_MBPS."""
    mbps = _get_network_mbps()
    decode_s = blob_bytes * _decode_ns_per_byte(fmt) / 1e9
    if mbps <= 0.0:
        return decode_s
    transfer_s = (blob_bytes * 8.0) / (mbps * 1_000_000.0)
    return transfer_s + decode_s


def _min_blob_format_latency(
    json_bytes: int,
    parquet_bytes: int,
    arrow_ipc_bytes: Optional[int],
) -> FormatName:
    """
    Minimize ACE-style score when FORMAT_LATENCY_NETWORK_MBPS > 0; else same as _min_blob_format.
    When MBPS == 0 but decode defaults matter, use bytes-only (backward compatible).
    """
    if _get_network_mbps() <= 0.0:
        return _min_blob_format(json_bytes, parquet_bytes, arrow_ipc_bytes)

    candidates: list[tuple[float, int, FormatName]] = [
        (_estimated_e2e_latency_seconds(json_bytes, "json"), 0, "json"),
        (_estimated_e2e_latency_seconds(parquet_bytes, "parquet_blob"), 1, "parquet_blob"),
    ]
    if arrow_ipc_bytes is not None:
        candidates.append(
            (_estimated_e2e_latency_seconds(arrow_ipc_bytes, "arrow_ipc_blob"), 2, "arrow_ipc_blob")
        )
    return min(candidates, key=lambda t: (t[0], t[1]))[2]


def select_format_with_hints(context: SelectionContext, hints: HintsDict) -> FormatName:
    """
    Choose format using optimization target and server-provided size hints.
    Falls back to select_format(context) if hints lack required keys.
    """
    # Unstructured payload selection (text/blob): uses different hint keys.
    if "raw_bytes" in hints or "text_inline_bytes" in hints:
        raw_b = hints.get("raw_bytes")
        if raw_b is None:
            # Can't do much without raw_bytes; fall back to json/parquet ladder.
            return select_format(context)
        raw_bytes = int(raw_b)
        gzip_b = hints.get("gzip_bytes")
        gzip_bytes = int(gzip_b) if gzip_b is not None else None
        inline_b = hints.get("text_inline_bytes")
        inline_bytes = int(inline_b) if inline_b is not None else None

        def min_unstructured_bytes() -> FormatName:
            candidates: list[tuple[int, int, FormatName]] = [(raw_bytes, 2, "raw_blob")]
            if gzip_bytes is not None:
                candidates.append((gzip_bytes, 1, "gzip_blob"))
            if inline_bytes is not None:
                candidates.append((inline_bytes, 0, "text_inline"))
            return min(candidates, key=lambda t: (t[0], t[1]))[2]

        if context.target == OptimizationTarget.MIN_BYTES:
            return min_unstructured_bytes()

        # For unstructured payloads, both min_latency and min_time_to_first_rows are
        # approximated as transfer-time dominated; decode is negligible (raw) or small (gzip).
        if _get_network_mbps() <= 0.0:
            return min_unstructured_bytes()

        def transfer_seconds(num_bytes: int) -> float:
            mbps = _get_network_mbps()
            if mbps <= 0.0:
                return float(num_bytes)
            return (num_bytes * 8.0) / (mbps * 1_000_000.0)

        candidates_s: list[tuple[float, int, FormatName]] = [
            (transfer_seconds(raw_bytes), 2, "raw_blob"),
        ]
        if gzip_bytes is not None:
            candidates_s.append((transfer_seconds(gzip_bytes), 1, "gzip_blob"))
        if inline_bytes is not None:
            candidates_s.append((transfer_seconds(inline_bytes), 0, "text_inline"))
        return min(candidates_s, key=lambda t: (t[0], t[1]))[2]

    json_b = hints.get("json_bytes")
    parquet_b = hints.get("parquet_bytes")
    if json_b is None or parquet_b is None:
        return select_format(context)
    json_bytes = int(json_b)
    parquet_bytes = int(parquet_b)
    arrow_ipc_b = hints.get("arrow_ipc_bytes")
    arrow_ipc_bytes = int(arrow_ipc_b) if arrow_ipc_b is not None else None

    first_chunk_b = hints.get("parquet_stream_first_chunk_bytes")
    first_chunk_bytes = int(first_chunk_b) if first_chunk_b is not None else None
    ipc_fc_b = hints.get("arrow_ipc_stream_first_chunk_bytes")
    arrow_ipc_first_chunk = int(ipc_fc_b) if ipc_fc_b is not None else None

    n_rows = context.n_rows
    target = context.target
    cells = n_rows * context.n_cols

    # Fix 4: JSON-obvious-winner short-circuit. For tiny payloads (the common
    # BIRD case) JSON is unambiguously the best choice; we skip looking at
    # parquet/IPC sizes entirely. MIN_TIME_TO_FIRST_ROWS still falls through
    # so streaming preference can override.
    if (
        target in (OptimizationTarget.MIN_LATENCY, OptimizationTarget.MIN_BYTES)
        and json_bytes <= _json_obvious_winner_bytes()
    ):
        return "json"

    if target == OptimizationTarget.MIN_BYTES:
        return _min_blob_format(json_bytes, parquet_bytes, arrow_ipc_bytes)

    if target == OptimizationTarget.MIN_TIME_TO_FIRST_ROWS:
        want_stream = context.prefer_streaming or n_rows >= ROWS_STREAM_FAVOR
        if want_stream and cells > CELLS_SMALL:
            if first_chunk_bytes is not None and arrow_ipc_first_chunk is not None:
                if _get_network_mbps() <= 0.0:
                    pick_pq = first_chunk_bytes <= arrow_ipc_first_chunk
                else:
                    pq_s = _estimated_e2e_latency_seconds(first_chunk_bytes, "parquet_blob")
                    ipc_s = _estimated_e2e_latency_seconds(arrow_ipc_first_chunk, "arrow_ipc_blob")
                    pick_pq = pq_s <= ipc_s
                return "parquet_stream" if pick_pq else "arrow_ipc_stream"
            if first_chunk_bytes is not None:
                return "parquet_stream"
            if arrow_ipc_first_chunk is not None:
                return "arrow_ipc_stream"
        return _min_blob_format_latency(json_bytes, parquet_bytes, arrow_ipc_bytes)

    # MIN_LATENCY: min-bytes proxy, or ACE-style transfer+decode when FORMAT_LATENCY_NETWORK_MBPS > 0.
    return _min_blob_format_latency(json_bytes, parquet_bytes, arrow_ipc_bytes)


def get_default_target() -> OptimizationTarget:
    """Default target from env FORMAT_SELECT_TARGET."""
    raw = os.environ.get("FORMAT_SELECT_TARGET", "min_latency").strip().lower()
    try:
        return OptimizationTarget(raw)
    except ValueError:
        return OptimizationTarget.MIN_LATENCY

