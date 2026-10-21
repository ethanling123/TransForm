"""
MCP + HTTP server for large-output benchmarking.

Run from the project root:

    uvicorn server_app:app --reload

This exposes:
- MCP control plane at:       http://localhost:8000/mcp
- Parquet blob data plane at: http://localhost:8000/blobs/{result_id}.parquet
- Parquet stream data plane at: http://localhost:8000/streams/{result_id}
- Arrow IPC blob data plane at: http://localhost:8000/ipc-blobs/{result_id}.arrow
- Arrow IPC stream data plane at: http://localhost:8000/ipc-streams/{result_id}
"""

from __future__ import annotations

import base64
import contextlib
import gzip
import hashlib
import json
import mimetypes
import os
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from functools import lru_cache
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.types import ASGIApp, Receive, Scope, Send

# Optional security/content libraries
try:  # python-magic for MIME detection
    import magic  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional dependency
    magic = None  # type: ignore[assignment]

try:  # PyJWT for session tokens
    import jwt  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional dependency
    jwt = None  # type: ignore[assignment]

try:  # Fernet for encryption
    from cryptography.fernet import Fernet  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional dependency
    Fernet = None  # type: ignore[assignment]

# Ensure we import the local python-sdk version of mcp
import sys

ROOT = Path(__file__).parent
# Where the server writes generated files (materialized results, benchmark outputs); defaults to the source folder.
SCRATCH = Path(os.environ.get("TRANSFORM_SCRATCH_DIR", ROOT))
PYTHON_SDK_SRC = ROOT / "python-sdk" / "src"
if str(PYTHON_SDK_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SDK_SRC))

from mcp.server.mcpserver import MCPServer  # type: ignore  # noqa: E402
from mcp.types import CallToolResult, TextContent  # type: ignore  # noqa: E402

import arms  # noqa: E402
import cost_model  # noqa: E402
from arms import REGISTRY  # noqa: E402
from format_selector import (  # noqa: E402
    OptimizationTarget,
    SelectionContext,
    _get_network_mbps,
    get_default_target,
    select_format_with_hints,
)
from format_mab import (  # noqa: E402
    load_mab_state,
    save_mab_state,
    select_format_with_mab,
    record_outcome,
)
from hint_reference_table import HintStore  # noqa: E402

if cost_model.enabled():
    cost_model.calibrate()  # a fraction of a second; done here so the first request does not pay for it


@dataclass
class ResultConfig:
    """Configuration for a benchmark dataset."""

    n_rows: int
    n_cols: int
    payload_kind: str = "tabular"  # "tabular" | "unstructured"
    arm: Optional[str] = None  # arm this handle was minted for (see arms.REGISTRY); None for registered results
    rows_per_chunk: int | None = None
    compression: str | None = None
    encoding_strategy: str | None = None
    ipc_compression: str | None = None
    materialized_path: Path | None = None
    raw_path: Path | None = None
    raw_mime_type: str | None = None
    raw_charset: str | None = None
    raw_gzip_path: Path | None = None
    # Compressed copies for the zstd/brotli arms, keyed by arm name (shared with derived configs).
    raw_codec_paths: Dict[str, Path] = field(default_factory=dict)
    # Pre-computed describe hints, populated at registration time so
    # describe_result_formats / large_result_auto are O(1) dict lookup
    # for the common (default codec, default rows_per_chunk) tuple.
    # Wrapper carries the codec/rows_per_chunk it was computed for so
    # non-default requests fall through to the live compute path.
    cached_hints: Optional[Dict[str, Any]] = None
    # Round-2 server payload cache (Fix F7). Each field is populated
    # eagerly at registration time when the body is in memory, then
    # consulted by the hot-path readers (`large_json`, parquet/IPC blob
    # endpoints, `_resolve_dataframe`). Bounded globally by
    # `RESULT_CACHE_MAX_BYTES` via `_PAYLOAD_CACHE_LRU`.
    cached_dataframe: Optional["pd.DataFrame"] = None
    cached_arrow_table: Optional["pa.Table"] = None
    cached_json_records: Optional[List[Dict[str, Any]]] = None
    cached_json_bytes: Optional[int] = None
    cached_parquet_blob_bytes: Optional[bytes] = None
    cached_arrow_ipc_blob_bytes: Optional[bytes] = None
    # Codec the cached blob bytes were encoded under (so requests for a
    # different codec fall through to a live encode).
    cached_parquet_codec: Optional[Tuple[str, str]] = None
    cached_arrow_ipc_codec: Optional[str] = None


# Simple in-memory registry keyed by UUIDs returned from tools.
RESULT_REGISTRY: Dict[str, ResultConfig] = {}


# Derived (per-format fetch) ids that share a source result's cached buffers, so evicting
# the source from the LRU also releases them.
_DERIVED_BY_SOURCE: Dict[str, List[str]] = {}


def _inherit_caches(dst_id: str, src_id: Optional[str]) -> None:
    """Share a registered result's primed payload caches with a derived per-format config.

    The large_* tools mint a fresh result id for each fetch URL; without this the data plane
    would re-read the Parquet file and re-encode it on every request.
    """
    src = RESULT_REGISTRY.get(src_id) if src_id is not None else None
    dst = RESULT_REGISTRY.get(dst_id)
    if src is None or dst is None:
        return
    dst.cached_dataframe = src.cached_dataframe
    dst.cached_arrow_table = src.cached_arrow_table
    dst.cached_parquet_blob_bytes = src.cached_parquet_blob_bytes
    dst.cached_parquet_codec = src.cached_parquet_codec
    dst.cached_arrow_ipc_blob_bytes = src.cached_arrow_ipc_blob_bytes
    dst.cached_arrow_ipc_codec = src.cached_arrow_ipc_codec
    _DERIVED_BY_SOURCE.setdefault(src_id, []).append(dst_id)

# Best-effort hints cache and reference store (for expensive encode-size estimates).
_HINTS_CACHE_BY_KEY_JSON: Dict[str, Dict[str, Any]] = {}
_HINTS_CACHE_MAX = 512
_HINT_STORE = HintStore.default()
_SERVER_MAB_STATE_PATH = Path(os.environ.get("FORMAT_MAB_STATE_PATH", "results/format_mab_state.json"))


# ----- Round-2 (F7): bounded payload cache for materialized result_ids ------------
# We keep an in-memory cache of the decoded DataFrame / Arrow table / encoded
# bytes for each materialized result_id, so the hot paths
# (`large_json`, `large_parquet_blob`, `large_arrow_ipc_blob`, the HTTP blob
# endpoints, and `describe_result_formats` via `_resolve_dataframe`) avoid
# `pd.read_parquet` and re-encoding on every request.
#
# Bounds: total cached bytes across all entries kept under
# `RESULT_CACHE_MAX_BYTES` (default 64 MB). When a registration would push
# the total past the cap, we evict the least recently used entries' cache
# fields (keeping the ResultConfig and its `cached_hints` so describe still
# works fast — only the bigger payload buffers are dropped).
_PAYLOAD_CACHE_LRU: "OrderedDict[str, int]" = OrderedDict()
_PAYLOAD_CACHE_TOTAL_BYTES: int = 0


def _result_cache_max_bytes() -> int:
    raw = os.environ.get("RESULT_CACHE_MAX_BYTES", "").strip()
    if raw:
        try:
            v = int(raw)
            return max(0, v)
        except ValueError:
            pass
    return 64 * 1024 * 1024  # 64 MB default


def _payload_cache_clear_entry(result_id: str) -> None:
    """Drop cached payload buffers (but keep ResultConfig + cached_hints)."""
    global _PAYLOAD_CACHE_TOTAL_BYTES
    cfg = RESULT_REGISTRY.get(result_id)
    prev = _PAYLOAD_CACHE_LRU.pop(result_id, 0)
    _PAYLOAD_CACHE_TOTAL_BYTES -= prev
    if _PAYLOAD_CACHE_TOTAL_BYTES < 0:
        _PAYLOAD_CACHE_TOTAL_BYTES = 0
    for target in [cfg] + [RESULT_REGISTRY.get(d) for d in _DERIVED_BY_SOURCE.pop(result_id, [])]:
        if target is not None:
            target.cached_dataframe = None
            target.cached_arrow_table = None
            target.cached_json_records = None
            target.cached_parquet_blob_bytes = None
            target.cached_arrow_ipc_blob_bytes = None
            target.cached_parquet_codec = None
            target.cached_arrow_ipc_codec = None


def _payload_cache_record(result_id: str, approx_bytes: int) -> None:
    """Mark a result_id as the most-recently-used payload cache entry."""
    global _PAYLOAD_CACHE_TOTAL_BYTES
    prev = _PAYLOAD_CACHE_LRU.pop(result_id, 0)
    _PAYLOAD_CACHE_TOTAL_BYTES -= prev
    _PAYLOAD_CACHE_LRU[result_id] = int(approx_bytes)
    _PAYLOAD_CACHE_TOTAL_BYTES += int(approx_bytes)
    cap = _result_cache_max_bytes()
    if cap <= 0:
        return
    while _PAYLOAD_CACHE_TOTAL_BYTES > cap and _PAYLOAD_CACHE_LRU:
        oldest_id, _ = next(iter(_PAYLOAD_CACHE_LRU.items()))
        if oldest_id == result_id:
            break  # don't evict the entry we just added
        _payload_cache_clear_entry(oldest_id)


def _payload_cache_touch(result_id: str) -> None:
    """Mark an existing entry as recently used without changing its size."""
    if result_id in _PAYLOAD_CACHE_LRU:
        _PAYLOAD_CACHE_LRU.move_to_end(result_id)


def _approx_payload_bytes(
    df: Optional["pd.DataFrame"],
    parquet_bytes: Optional[bytes],
    ipc_bytes: Optional[bytes],
    json_records: Optional[List[Dict[str, Any]]],
) -> int:
    n = 0
    if df is not None:
        try:
            n += int(df.memory_usage(deep=True).sum())
        except Exception:
            pass
    if parquet_bytes is not None:
        n += len(parquet_bytes)
    if ipc_bytes is not None:
        n += len(ipc_bytes)
    if json_records is not None:
        # Cheap upper-ish bound: 64 B per cell.
        n += len(json_records) * (len(json_records[0]) if json_records else 0) * 64
    return n


def _populate_materialized_caches(
    cfg: ResultConfig,
    *,
    result_id: str,
    df: pd.DataFrame,
    table: Optional[pa.Table] = None,
    parquet_bytes: Optional[bytes] = None,
    rows_per_chunk: int = 8192,
) -> None:
    """
    F2 + F7 + F10: at registration time, populate `cached_hints` and the
    payload caches (DataFrame, Arrow table, JSON records, encoded blob bytes)
    for the default codec tuple, then record the entry size in the LRU.

    Called from both `register_materialized_endpoint` (POST /materialized)
    and `bird_query_materialize` so every materialized result_id starts
    primed. Best-effort: any single sub-step failure leaves that cache
    field as `None` and falls back to the live compute path.
    """
    default_comp = _get_default_compression()
    default_enc_strat = _get_default_encoding_strategy()
    default_ipc_comp = _get_default_arrow_ipc_compression()

    if table is None:
        try:
            table = pa.Table.from_pandas(df, preserve_index=False)
        except Exception:
            table = None

    # F7: store DataFrame + Arrow table.
    cfg.cached_dataframe = df
    cfg.cached_arrow_table = table

    # F2: hints (re-uses table when present so we don't re-encode).
    try:
        hints = _compute_tabular_size_hints_from_df(
            df,
            rows_per_chunk=rows_per_chunk,
            comp=default_comp,
            enc_strat=default_enc_strat,
            ipc_comp=default_ipc_comp,
            table=table,
        )
        cfg.cached_hints = {
            "rows_per_chunk": int(rows_per_chunk),
            "parquet_compression": default_comp,
            "parquet_encoding_strategy": default_enc_strat,
            "arrow_ipc_compression": default_ipc_comp,
            "hints": hints,
        }
    except Exception:
        cfg.cached_hints = None

    # F7: cache JSON records when the payload is small enough that storing
    # them is cheap (this is the BIRD common case: <= SMALL_PAYLOAD_CELLS).
    try:
        n_rows = len(df)
        n_cols = len(df.columns)
        if n_rows * n_cols <= SMALL_PAYLOAD_CELLS:
            cfg.cached_json_records = df.to_dict(orient="records")
            try:
                cfg.cached_json_bytes = len(
                    json.dumps(cfg.cached_json_records).encode("utf-8")
                )
            except Exception:
                cfg.cached_json_bytes = None
    except Exception:
        cfg.cached_json_records = None

    # F7: cache encoded Parquet bytes for the default codec. If the caller
    # already had encoded bytes (e.g., bird_query_materialize re-uses the
    # bytes it just wrote to disk), prefer those.
    try:
        if parquet_bytes is not None:
            cfg.cached_parquet_blob_bytes = parquet_bytes
            cfg.cached_parquet_codec = (default_comp, default_enc_strat)
        elif table is not None:
            enc = _encode_parquet(table, default_comp, default_enc_strat)
            cfg.cached_parquet_blob_bytes = enc
            cfg.cached_parquet_codec = (default_comp, default_enc_strat)
    except Exception:
        cfg.cached_parquet_blob_bytes = None
        cfg.cached_parquet_codec = None

    # F7: cache encoded Arrow IPC bytes for the default codec.
    try:
        if table is not None:
            cfg.cached_arrow_ipc_blob_bytes = _encode_arrow_ipc_file(
                table, default_ipc_comp,
            )
            cfg.cached_arrow_ipc_codec = default_ipc_comp
    except Exception:
        cfg.cached_arrow_ipc_blob_bytes = None
        cfg.cached_arrow_ipc_codec = None

    # Record approximate memory footprint and evict LRU if needed.
    approx = _approx_payload_bytes(
        cfg.cached_dataframe,
        cfg.cached_parquet_blob_bytes,
        cfg.cached_arrow_ipc_blob_bytes,
        cfg.cached_json_records,
    )
    _payload_cache_record(result_id, approx)


def _hints_cache_get(key_json: str) -> Optional[Dict[str, Any]]:
    return _HINTS_CACHE_BY_KEY_JSON.get(key_json)


def _hints_cache_put(key_json: str, hints: Dict[str, Any]) -> None:
    if key_json in _HINTS_CACHE_BY_KEY_JSON:
        _HINTS_CACHE_BY_KEY_JSON[key_json] = hints
        return
    if len(_HINTS_CACHE_BY_KEY_JSON) >= _HINTS_CACHE_MAX:
        # Simple eviction: drop one arbitrary entry (good enough for benchmark server).
        _HINTS_CACHE_BY_KEY_JSON.pop(next(iter(_HINTS_CACHE_BY_KEY_JSON)))
    _HINTS_CACHE_BY_KEY_JSON[key_json] = hints


def _hints_db_disabled() -> bool:
    return os.environ.get("FORMAT_HINTS_DB_DISABLE", "").strip().lower() in {"1", "true", "yes"}


def _tabular_hints_key(
    *,
    kind: str,
    n_rows: int,
    n_cols: int,
    rows_per_chunk: int,
    parquet_compression: str,
    parquet_encoding_strategy: str,
    arrow_ipc_compression: str,
    result_id: Optional[str],
    materialized_path: Optional[Path],
) -> Dict[str, Any]:
    """
    Stable key for looking up cached/precomputed tabular format size hints.
    """
    key: Dict[str, Any] = {
        "kind": kind,
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "rows_per_chunk": int(rows_per_chunk),
        "parquet_compression": parquet_compression,
        "parquet_encoding_strategy": parquet_encoding_strategy,
        "arrow_ipc_compression": arrow_ipc_compression,
    }
    if result_id is not None:
        key["result_id"] = result_id
    if materialized_path is not None and materialized_path.is_file():
        st = materialized_path.stat()
        key["materialized_path"] = str(materialized_path)
        key["materialized_mtime_ns"] = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        key["materialized_size"] = int(st.st_size)
    return key


def _stable_key_json(key: Dict[str, Any]) -> str:
    return json.dumps(key, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

# In-memory registry of validated files keyed by validation id
VALIDATED_FILES: Dict[str, Dict[str, Any]] = {}

# In-memory JWT-style session store: token -> username
ACTIVE_SESSIONS: Dict[str, str] = {}

# Directory for uploaded Parquet files registered via POST /materialized
MATERIALIZED_DIR = SCRATCH / "data" / "materialized"
MATERIALIZED_RAW_DIR = SCRATCH / "data" / "materialized_raw"
MAX_MATERIALIZED_ROWS = 1_000_000
MAX_JSON_CELLS = 5_000_000
MAX_INLINE_TEXT_BYTES = 256 * 1024  # inline unstructured payload cap (MCP structured_content)


def _json_cells_cap() -> Optional[int]:
    """
    JSON inline cap (cells = rows*cols) for baseline-like tools.

    Benchmark-only override:
      - DISABLE_JSON_CAP=1 disables the cap entirely
      - MAX_JSON_CELLS_OVERRIDE=<int> sets a new cap
    """
    raw_disable = os.environ.get("DISABLE_JSON_CAP", "").strip().lower()
    if raw_disable in {"1", "true", "yes", "y"}:
        return None
    raw = os.environ.get("MAX_JSON_CELLS_OVERRIDE", "").strip()
    if raw:
        try:
            v = int(raw)
            return v if v > 0 else None
        except ValueError:
            pass
    return MAX_JSON_CELLS


# ----- File validation helpers ------------------------------------------------------

ALLOWED_MIME_TYPES: Set[str] = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "audio/wav",
    "audio/mp3",
    "video/mp4",
    "application/pdf",
}

MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB


def _get_file_size(path: Path) -> int:
    return path.stat().st_size


def _read_file_header(path: Path, num_bytes: int) -> bytes:
    with path.open("rb") as f:
        return f.read(num_bytes)


def _calculate_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _detect_mime_type(path: Path, header: bytes) -> str:
    if magic is not None:
        try:
            detected = magic.from_buffer(header, mime=True)
            if isinstance(detected, str):
                return detected
        except Exception:
            pass

    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def _validate_file_locally(file_path: str, expected_type: Optional[str]) -> Dict[str, Any]:
    """
    Blog-style comprehensive file validation:
    - existence and size check
    - MIME validation against allowed list
    - optional exact type match
    - SHA-256 hash summary
    """

    path = Path(file_path)
    if not path.is_file():
        return {"valid": False, "error": f"File not found: {file_path}"}

    size = _get_file_size(path)
    if size > MAX_FILE_SIZE_BYTES:
        return {
            "valid": False,
            "error": f"File size {size} exceeds limit {MAX_FILE_SIZE_BYTES}",
        }

    header = _read_file_header(path, 2048)
    detected_mime = _detect_mime_type(path, header)

    if detected_mime not in ALLOWED_MIME_TYPES:
        return {"valid": False, "error": f"MIME type {detected_mime} not allowed"}

    if expected_type and detected_mime != expected_type:
        return {
            "valid": False,
            "error": f"MIME type mismatch: expected {expected_type}, got {detected_mime}",
        }

    file_hash = _calculate_file_hash(path)
    return {
        "valid": True,
        "details": f"Type: {detected_mime}, Size: {size}, Hash: {file_hash[:16]}",
        "mime": detected_mime,
        "size": size,
        "hash": file_hash,
        "file_path": str(path),
    }


TPCDS_PARQUET_PATH = os.environ.get("TPCDS_PARQUET_PATH")


@lru_cache(maxsize=1)
def _load_tpcds_base_table() -> pd.DataFrame:
    """
    Lazily load a base TPC-DS table from Parquet when TPCDS_PARQUET_PATH is set.

    This lets us back benchmarks with a real analytical schema (e.g. catalog_sales)
    without changing the MCP tool interface. For large tables, we rely on Parquet's
    columnar layout and only slice the needed rows/columns for each run.
    """

    if not TPCDS_PARQUET_PATH:
        raise RuntimeError("TPCDS_PARQUET_PATH is not set but _load_tpcds_base_table was called")
    path = Path(TPCDS_PARQUET_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"TPCDS_PARQUET_PATH points to missing file: {path}")
    return pd.read_parquet(path)


def _generate_dataframe(n_rows: int, n_cols: int, *, offset: int = 0) -> pd.DataFrame:
    """
    Generate a benchmarking dataframe.

    - If TPCDS_PARQUET_PATH is set, we draw a slice from that real table
      (e.g. TPC-DS catalog_sales) to back all three modes (JSON, blob, stream).
    - Otherwise, fall back to a deterministic synthetic numeric dataframe.
    """

    if TPCDS_PARQUET_PATH:
        base = _load_tpcds_base_table()
        # Ensure we don't go out of bounds; for very large requested sizes the caller
        # should choose n_rows based on the underlying table.
        start = max(offset, 0)
        stop = min(start + n_rows, len(base))
        if start >= len(base):
            # Empty slice if offset is beyond table size
            df = base.iloc[0:0].copy()
        else:
            df = base.iloc[start:stop].copy()

        # Select the first n_cols columns to keep the same interface semantics.
        # If the underlying table has fewer columns than requested, we just return all.
        if n_cols > 0:
            cols = list(df.columns)
            if len(cols) > n_cols:
                cols = cols[:n_cols]
            df = df[cols]
        return df

    # Synthetic fallback: deterministic numeric dataframe
    index = np.arange(offset, offset + n_rows, dtype=np.int64)
    data: Dict[str, Any] = {"row_id": index}

    for j in range(n_cols):
        # Simple column pattern: linear + some variation
        data[f"col_{j}"] = index * (j + 1) + j

    return pd.DataFrame(data)


def _load_materialized_dataframe(
    path: Path, *, offset: int = 0, limit: int | None = None,
) -> pd.DataFrame:
    """Load a registered materialized Parquet file into a DataFrame."""
    df = pd.read_parquet(path)
    if offset > 0:
        df = df.iloc[offset:]
    if limit is not None:
        df = df.iloc[:limit]
    return df


def _resolve_dataframe(
    config: ResultConfig, *, offset: int = 0, limit: int | None = None,
) -> pd.DataFrame:
    """Return the DataFrame backing a ResultConfig — materialized or generated.

    Round-2 (F7): when the materialized DataFrame is cached on the
    `ResultConfig` (populated at registration), serve from cache and skip
    the `pd.read_parquet` round-trip. Slicing for offset/limit happens on
    the cached DataFrame.
    """
    if config.cached_dataframe is not None:
        df = config.cached_dataframe
        if offset > 0 or limit is not None:
            stop = (offset + limit) if limit is not None else None
            df = df.iloc[offset:stop]
        return df
    if config.materialized_path is not None:
        return _load_materialized_dataframe(
            config.materialized_path, offset=offset, limit=limit,
        )
    n = limit if limit is not None else config.n_rows
    return _generate_dataframe(n_rows=n, n_cols=config.n_cols, offset=offset)


def _resolve_arrow_table(config: ResultConfig) -> pa.Table:
    """Return the Arrow table backing a materialized ResultConfig (cache-first)."""
    if config.cached_arrow_table is not None:
        return config.cached_arrow_table
    df = _resolve_dataframe(config)
    return pa.Table.from_pandas(df, preserve_index=False)


# ----- Parquet compression / encoding config ----------------------------------------

VALID_COMPRESSIONS = {"snappy", "gzip", "zstd", "brotli", "lz4", "none"}
VALID_ENCODING_STRATEGIES = {"default", "data_driven"}


def _get_default_compression() -> str:
    """Read PARQUET_COMPRESSION env var; default 'zstd'."""
    raw = os.environ.get("PARQUET_COMPRESSION", "zstd").strip().lower()
    return raw if raw in VALID_COMPRESSIONS else "zstd"


def _get_default_encoding_strategy() -> str:
    """Read PARQUET_ENCODING_STRATEGY env var; default 'default'."""
    raw = os.environ.get("PARQUET_ENCODING_STRATEGY", "default").strip().lower()
    return raw if raw in VALID_ENCODING_STRATEGIES else "default"


def _encode_parquet(table: pa.Table, compression: str, encoding_strategy: str, params: Optional[Dict[str, Any]] = None) -> bytes:
    """Encode an Arrow table to Parquet bytes with the given codec and strategy."""
    return arms.encode_parquet(table, compression, None, encoding_strategy, params)


# ----- Arrow IPC (file format) ----------------------------------------------------

VALID_ARROW_IPC_COMPRESSIONS = {"none", "lz4", "zstd"}


def _get_default_arrow_ipc_compression() -> str:
    """Read ARROW_IPC_COMPRESSION env var; default 'zstd'."""
    raw = os.environ.get("ARROW_IPC_COMPRESSION", "zstd").strip().lower()
    return raw if raw in VALID_ARROW_IPC_COMPRESSIONS else "zstd"


def _encode_arrow_ipc_file(table: pa.Table, ipc_compression: str) -> bytes:
    """Encode an Arrow table as IPC file bytes (single file with schema + batches)."""
    return arms.encode_arrow_ipc(table, ipc_compression)


# ----- Parquet encoding helpers with caching ----------------------------------------


@lru_cache(maxsize=128)
def _get_parquet_blob_bytes(
    n_rows: int,
    n_cols: int,
    compression: str = "snappy",
    encoding_strategy: str = "default",
) -> bytes:
    """
    Return Parquet-encoded blob bytes for the full dataset.

    Cached by (n_rows, n_cols, compression, encoding_strategy).
    """
    df = _generate_dataframe(n_rows=n_rows, n_cols=n_cols)
    table = pa.Table.from_pandas(df, preserve_index=False)
    return _encode_parquet(table, compression, encoding_strategy)


@lru_cache(maxsize=1024)
def _get_parquet_chunk_bytes(
    n_rows: int,
    n_cols: int,
    offset: int,
    this_rows: int,
    compression: str = "snappy",
    encoding_strategy: str = "default",
) -> bytes:
    """
    Return Parquet-encoded chunk bytes for a slice of the dataset.

    Cached by (n_rows, n_cols, offset, this_rows, compression, encoding_strategy).
    """
    df = _generate_dataframe(n_rows=this_rows, n_cols=n_cols, offset=offset)
    table = pa.Table.from_pandas(df, preserve_index=False)
    return _encode_parquet(table, compression, encoding_strategy)


@lru_cache(maxsize=128)
def _get_arrow_ipc_blob_bytes(
    n_rows: int,
    n_cols: int,
    ipc_compression: str = "none",
) -> bytes:
    """Return IPC file bytes for the full dataset (cached)."""
    df = _generate_dataframe(n_rows=n_rows, n_cols=n_cols)
    table = pa.Table.from_pandas(df, preserve_index=False)
    return _encode_arrow_ipc_file(table, ipc_compression)


@lru_cache(maxsize=1024)
def _get_arrow_ipc_chunk_bytes(
    n_rows: int,
    n_cols: int,
    offset: int,
    this_rows: int,
    ipc_compression: str = "none",
) -> bytes:
    """Return IPC file bytes for one row slice (cached)."""
    df = _generate_dataframe(n_rows=this_rows, n_cols=n_cols, offset=offset)
    table = pa.Table.from_pandas(df, preserve_index=False)
    return _encode_arrow_ipc_file(table, ipc_compression)


# Threshold (in cells = n_rows * n_cols) below which we always compute exact
# JSON byte size with json.dumps. Above it we use a fast estimator (Fix 3).
SMALL_PAYLOAD_CELLS = 4096

# Hints byte threshold: when materialized JSON is at or below this, the format
# selector can pick "json" without ever looking at parquet/IPC sizes (Fix 4).
# Override via env FORMAT_HINTS_JSON_OBVIOUS_WINNER_BYTES.
JSON_OBVIOUS_WINNER_BYTES_DEFAULT = 4096


def _json_obvious_winner_bytes() -> int:
    raw = os.environ.get("FORMAT_HINTS_JSON_OBVIOUS_WINNER_BYTES", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return JSON_OBVIOUS_WINNER_BYTES_DEFAULT


def _format_hints_skip_large_for_small() -> bool:
    """When True (default), skip Parquet+IPC encodes if JSON is the obvious winner."""
    raw = os.environ.get("FORMAT_HINTS_SKIP_LARGE_FORMATS_FOR_SMALL", "1").strip().lower()
    return raw not in {"0", "false", "no"}


# Per-dtype JSON value width budgets used by the estimator. Slightly conservative
# (over-estimate side) so the selector never under-counts JSON; biasing toward
# the slower-but-smaller format is safer than biasing away from JSON when JSON
# is in fact much smaller.
_JSON_INT_VALUE_WIDTH = 12       # signed 32-bit ints serialise to <=11 chars + comma
_JSON_FLOAT_VALUE_WIDTH = 22     # repr of a 64-bit float
_JSON_BOOL_VALUE_WIDTH = 5       # "false"
_JSON_DATETIME_VALUE_WIDTH = 30  # ISO-8601 with microseconds + quotes
_JSON_DEFAULT_VALUE_WIDTH = 32   # fallback for unknown dtypes


def _estimate_json_bytes_from_df(df: pd.DataFrame) -> int:
    """
    Approximate len(json.dumps(df.to_dict(orient='records')).encode('utf-8')).

    Conservative (slight over-estimate) per-dtype width formula. For string /
    object columns we sample the actual character count via
    ``df[col].astype(str).str.len()`` so estimates do not collapse on big
    text columns. Used above ``SMALL_PAYLOAD_CELLS``.
    """
    n_rows = len(df)
    if n_rows == 0:
        return 2  # "[]"
    per_record_overhead = 2  # "{}"
    record_value_bytes = 0
    for col in df.columns:
        key_overhead = len(str(col)) + 4  # "key":<value>,
        per_record_overhead += key_overhead
        dtype = df[col].dtype
        if pd.api.types.is_bool_dtype(dtype):
            record_value_bytes += _JSON_BOOL_VALUE_WIDTH
        elif pd.api.types.is_integer_dtype(dtype):
            record_value_bytes += _JSON_INT_VALUE_WIDTH
        elif pd.api.types.is_float_dtype(dtype):
            record_value_bytes += _JSON_FLOAT_VALUE_WIDTH
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            record_value_bytes += _JSON_DATETIME_VALUE_WIDTH
        else:
            # object/string: actual data size + 2 quotes per value.
            try:
                col_chars = int(df[col].astype(str).str.len().sum())
                record_value_bytes += col_chars // max(1, n_rows) + 2
            except Exception:
                record_value_bytes += _JSON_DEFAULT_VALUE_WIDTH
    per_record = per_record_overhead + record_value_bytes
    # array brackets + per-record bytes + n_rows-1 commas between records
    return 2 + n_rows * per_record + max(0, n_rows - 1)


def _measure_json_bytes_from_df(df: pd.DataFrame) -> int:
    """Exact JSON byte size for small payloads; estimator for larger ones."""
    n_rows = len(df)
    n_cols = len(df.columns)
    if n_rows * n_cols <= SMALL_PAYLOAD_CELLS:
        records = df.to_dict(orient="records")
        return len(json.dumps(records).encode("utf-8"))
    return _estimate_json_bytes_from_df(df)


@lru_cache(maxsize=128)
def _get_json_byte_size(n_rows: int, n_cols: int) -> int:
    """
    Return the byte size of the synthetic dataset as JSON (same encoding as
    large_json). Cached so hint tool stays cheap for repeated (n_rows, n_cols).

    Used only on the synthetic (no-result_id) path; for materialized results
    we go through ``_measure_json_bytes_from_df`` so we benefit from the
    estimator on large tables too.
    """
    df = _generate_dataframe(n_rows=n_rows, n_cols=n_cols)
    return _measure_json_bytes_from_df(df)


def _compute_tabular_size_hints_from_df(
    df: pd.DataFrame,
    *,
    rows_per_chunk: int,
    comp: str,
    enc_strat: str,
    ipc_comp: str,
    table: Optional[pa.Table] = None,
) -> Dict[str, Any]:
    """
    Compute size hints from a concrete DataFrame (and optional pre-built Arrow
    table). When the JSON byte size is at or below the obvious-winner threshold
    AND ``FORMAT_HINTS_SKIP_LARGE_FORMATS_FOR_SMALL`` is enabled, skip the
    Parquet and Arrow IPC encodes entirely and return placeholder large bytes
    so the format selector still picks JSON (Fix 4).
    """
    n_rows = len(df)
    n_cols = len(df.columns)

    json_bytes = _measure_json_bytes_from_df(df)

    skip_large = (
        _format_hints_skip_large_for_small()
        and json_bytes <= _json_obvious_winner_bytes()
    )

    if skip_large:
        # Return placeholder values: large enough that the selector picks JSON
        # under MIN_BYTES / MIN_LATENCY without any real Parquet/IPC encode.
        sentinel = max(json_bytes * 64, 1 << 30)
        return {
            "resolved_n_rows": n_rows,
            "resolved_n_cols": n_cols,
            "json_bytes": int(json_bytes),
            "parquet_bytes": int(sentinel),
            "parquet_stream_first_chunk_bytes": int(sentinel),
            "arrow_ipc_bytes": int(sentinel),
            "arrow_ipc_stream_first_chunk_bytes": int(sentinel),
            "small_payload_skip_large_formats": True,
        }

    if table is None:
        table = pa.Table.from_pandas(df, preserve_index=False)
    parquet_bytes = len(_encode_parquet(table, comp, enc_strat))
    arrow_ipc_bytes = len(_encode_arrow_ipc_file(table, ipc_comp))
    first_chunk_rows = min(rows_per_chunk, n_rows)
    if first_chunk_rows > 0:
        chunk_table = table.slice(0, first_chunk_rows)
        parquet_stream_first_chunk_bytes = len(
            _encode_parquet(chunk_table, comp, enc_strat)
        )
        arrow_ipc_stream_first_chunk_bytes = len(
            _encode_arrow_ipc_file(chunk_table, ipc_comp)
        )
    else:
        parquet_stream_first_chunk_bytes = 0
        arrow_ipc_stream_first_chunk_bytes = 0
    return {
        "resolved_n_rows": n_rows,
        "resolved_n_cols": n_cols,
        "json_bytes": int(json_bytes),
        "parquet_bytes": int(parquet_bytes),
        "parquet_stream_first_chunk_bytes": int(parquet_stream_first_chunk_bytes),
        "arrow_ipc_bytes": int(arrow_ipc_bytes),
        "arrow_ipc_stream_first_chunk_bytes": int(arrow_ipc_stream_first_chunk_bytes),
    }


def _compute_tabular_size_hints(
    *,
    n_rows: int,
    n_cols: int,
    rows_per_chunk: int,
    result_id: Optional[str],
    comp: str,
    enc_strat: str,
    ipc_comp: str,
) -> Dict[str, Any]:
    """
    Compute the same size hints as describe_result_formats.
    This may do real encodes; callers should cache via reference table when possible.
    """
    if result_id is not None:
        config = RESULT_REGISTRY.get(result_id)
        if config is None:
            raise ValueError(f"Unknown result_id: {result_id}")
        df = _resolve_dataframe(config)
        return _compute_tabular_size_hints_from_df(
            df,
            rows_per_chunk=rows_per_chunk,
            comp=comp,
            enc_strat=enc_strat,
            ipc_comp=ipc_comp,
        )

    # Synthetic (no result_id) path: keep the lru_cache helpers since shapes
    # repeat across requests and reuse is cheap.
    json_bytes = _get_json_byte_size(n_rows, n_cols)
    if (
        _format_hints_skip_large_for_small()
        and json_bytes <= _json_obvious_winner_bytes()
    ):
        sentinel = max(json_bytes * 64, 1 << 30)
        return {
            "resolved_n_rows": n_rows,
            "resolved_n_cols": n_cols,
            "json_bytes": int(json_bytes),
            "parquet_bytes": int(sentinel),
            "parquet_stream_first_chunk_bytes": int(sentinel),
            "arrow_ipc_bytes": int(sentinel),
            "arrow_ipc_stream_first_chunk_bytes": int(sentinel),
            "small_payload_skip_large_formats": True,
        }
    parquet_bytes = len(_get_parquet_blob_bytes(n_rows, n_cols, comp, enc_strat))
    arrow_ipc_bytes = len(_get_arrow_ipc_blob_bytes(n_rows, n_cols, ipc_comp))
    first_chunk_rows = min(rows_per_chunk, n_rows)
    parquet_stream_first_chunk_bytes = len(
        _get_parquet_chunk_bytes(
            n_rows, n_cols, 0, first_chunk_rows, comp, enc_strat,
        )
    )
    arrow_ipc_stream_first_chunk_bytes = len(
        _get_arrow_ipc_chunk_bytes(
            n_rows, n_cols, 0, first_chunk_rows, ipc_comp,
        )
    )
    return {
        "resolved_n_rows": n_rows,
        "resolved_n_cols": n_cols,
        "json_bytes": json_bytes,
        "parquet_bytes": parquet_bytes,
        "parquet_stream_first_chunk_bytes": parquet_stream_first_chunk_bytes,
        "arrow_ipc_bytes": arrow_ipc_bytes,
        "arrow_ipc_stream_first_chunk_bytes": arrow_ipc_stream_first_chunk_bytes,
    }


def _cached_hints_match(
    cached: Optional[Dict[str, Any]],
    *,
    rows_per_chunk: int,
    comp: str,
    enc_strat: str,
    ipc_comp: str,
    resolved_n_rows: int,
) -> bool:
    """
    Verify a ResultConfig.cached_hints wrapper matches the request codec/chunk.

    ``rows_per_chunk`` is only meaningful through the first-chunk row count
    ``min(rows_per_chunk, n_rows)``. The BIRD bench passes
    ``min(8192, n_rows)`` while registration may pre-compute with a different
    declared chunk as long as the effective first chunk matches.
    """
    if not isinstance(cached, dict):
        return False
    if not isinstance(cached.get("hints"), dict):
        return False
    if (
        cached.get("parquet_compression") != comp
        or cached.get("parquet_encoding_strategy") != enc_strat
        or cached.get("arrow_ipc_compression") != ipc_comp
    ):
        return False
    n_rows = max(1, int(resolved_n_rows))
    eff_cached = min(int(cached.get("rows_per_chunk") or 0), n_rows)
    eff_req = min(int(rows_per_chunk), n_rows)
    return eff_cached == eff_req


def _get_tabular_size_hints_cached(
    *,
    n_rows: int,
    n_cols: int,
    rows_per_chunk: int,
    result_id: Optional[str],
    comp: str,
    enc_strat: str,
    ipc_comp: str,
) -> Dict[str, Any]:
    """
    Cache → compute.

    For result_id requests we go straight through the in-memory
    ResultConfig.cached_hints (populated at registration). The persisted
    SQLite HintStore is bypassed for result_id since each result_id is
    process-local; reopening a fresh SQLite connection per call is pure
    overhead (Fix 1).

    For synthetic (no result_id) shapes we keep the in-process key-json
    cache and the SQLite store, since identical shapes recur across runs.
    """
    if result_id is not None:
        cfg = RESULT_REGISTRY.get(result_id)
        resolved_n_rows = int(cfg.n_rows) if cfg is not None else int(n_rows)
        if cfg is not None and _cached_hints_match(
            cfg.cached_hints,
            rows_per_chunk=rows_per_chunk,
            comp=comp,
            enc_strat=enc_strat,
            ipc_comp=ipc_comp,
            resolved_n_rows=resolved_n_rows,
        ):
            return cfg.cached_hints["hints"]  # type: ignore[index]

        # Cache miss (non-default codec/rows_per_chunk, or pre-fix registration).
        computed = _compute_tabular_size_hints(
            n_rows=n_rows,
            n_cols=n_cols,
            rows_per_chunk=rows_per_chunk,
            result_id=result_id,
            comp=comp,
            enc_strat=enc_strat,
            ipc_comp=ipc_comp,
        )
        # Stash for repeated requests with the same codec/rows_per_chunk.
        if cfg is not None and cfg.cached_hints is None:
            cfg.cached_hints = {
                "rows_per_chunk": int(rows_per_chunk),
                "parquet_compression": comp,
                "parquet_encoding_strategy": enc_strat,
                "arrow_ipc_compression": ipc_comp,
                "hints": computed,
            }
        return computed

    # Synthetic (no result_id) path: keep the existing in-memory + SQLite caches.
    key = _tabular_hints_key(
        kind="tabular",
        n_rows=n_rows,
        n_cols=n_cols,
        rows_per_chunk=rows_per_chunk,
        parquet_compression=comp,
        parquet_encoding_strategy=enc_strat,
        arrow_ipc_compression=ipc_comp,
        result_id=None,
        materialized_path=None,
    )
    key_json = _stable_key_json(key)

    cached = _hints_cache_get(key_json)
    if isinstance(cached, dict):
        return cached

    if not _hints_db_disabled():
        stored = _HINT_STORE.get(key)
        if isinstance(stored, dict):
            _hints_cache_put(key_json, stored)
            return stored

    computed = _compute_tabular_size_hints(
        n_rows=n_rows,
        n_cols=n_cols,
        rows_per_chunk=rows_per_chunk,
        result_id=None,
        comp=comp,
        enc_strat=enc_strat,
        ipc_comp=ipc_comp,
    )
    _hints_cache_put(key_json, computed)
    if not _hints_db_disabled():
        _HINT_STORE.upsert(key, computed)
    return computed

# ----- MCP server (control plane) -------------------------------------------------

mcp = MCPServer(name="LargeOutputBenchmark")


# ----- BIRD SQL execution (server-side, for end-to-end benchmarks) -----------------

def _resolve_bird_sqlite_path(db_id: str) -> Optional[Path]:
    """
    Resolve <db_id>.sqlite using layouts used by BIRD mini-dev.

    Looks in:
      - $BIRD_SQLITE_ROOT/dev_databases/<db_id>/<db_id>.sqlite
      - data/datasets/bird/dev/dev_databases/<db_id>/<db_id>.sqlite
      - data/datasets/bird/dev/databases/<db_id>/<db_id>.sqlite
    """
    db_id = (db_id or "").strip()
    if not db_id:
        return None
    name = f"{db_id}.sqlite"
    roots: list[Path] = []
    env = os.environ.get("BIRD_SQLITE_ROOT", "").strip()
    if env:
        roots.append(Path(env) / "dev_databases" / db_id / name)
    base = Path("data/datasets/bird/dev")
    roots.extend(
        [
            base / "dev_databases" / db_id / name,
            base / "databases" / db_id / name,
        ]
    )
    for p in roots:
        if p.is_file():
            return p
    return None


def bird_sql_for_sqlite(sql: str) -> str:
    """BIRD gold SQL uses MySQL-style backticks; SQLite expects double quotes."""
    return (sql or "").replace("`", '"')


def _execute_sqlite_query_to_df(db_path: Path, sql: str, max_rows: int) -> pd.DataFrame:
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    # Some benchmark databases (e.g. Spider) hold text that is not valid UTF-8.
    conn.text_factory = lambda raw: raw.decode("utf-8", errors="replace")
    try:
        # Prevent pathological queries from stalling full-dev benchmark runs.
        # This uses SQLite's progress handler to interrupt long-running queries.
        raw_timeout = os.environ.get("SQLITE_QUERY_TIMEOUT_S", "30").strip()
        try:
            timeout_s = float(raw_timeout)
        except ValueError:
            timeout_s = 30.0
        if timeout_s > 0:
            t0 = time.perf_counter()

            def _progress_handler() -> int:
                return 1 if (time.perf_counter() - t0) > timeout_s else 0

            # Called every N virtual machine instructions.
            conn.set_progress_handler(_progress_handler, 10_000)

        df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()
    # Parquet/Arrow reject duplicate column names and JSON records would silently drop one.
    seen: Dict[str, int] = {}
    unique_names = []
    for name in df.columns:
        count = seen.get(name, 0)
        seen[name] = count + 1
        unique_names.append(name if count == 0 else f"{name}_{count}")
    df.columns = unique_names
    # SQLite columns can mix types (e.g. ints and strings); Arrow needs one type per column.
    for col in df.columns:
        if df[col].dtype == object and len({type(v) for v in df[col].dropna()}) > 1:
            df[col] = df[col].astype(str)
    if max_rows > 0 and len(df) > max_rows:
        df = df.iloc[:max_rows].copy()
    return df


@mcp.tool()
def bird_query_json(
    db_id: str,
    sql: str,
    max_rows: int = 500_000,
) -> List[Dict[str, Any]]:
    """
    Execute BIRD SQL on the server and return inline JSON records (baseline for E2E).
    """
    db_path = _resolve_bird_sqlite_path(db_id)
    if db_path is None:
        raise ValueError(f"Could not resolve BIRD sqlite for db_id={db_id!r}. Set BIRD_SQLITE_ROOT.")
    df = _execute_sqlite_query_to_df(db_path, bird_sql_for_sqlite(sql), max_rows=max_rows)
    cells = len(df) * len(df.columns)
    cap = _json_cells_cap()
    if cap is not None and cells > cap:
        raise ValueError(f"Result too large for JSON ({cells} cells > {cap})")
    return df.to_dict(orient="records")


@mcp.tool()
def bird_query_materialize(
    db_id: str,
    sql: str,
    max_rows: int = 500_000,
) -> Dict[str, Any]:
    """
    Execute BIRD SQL on the server, materialize to Parquet, and return a result_id.

    This is a helper for format-selection benchmarking where subsequent tools
    (describe_result_formats / large_* / large_result_auto) operate on result_id.
    """
    db_path = _resolve_bird_sqlite_path(db_id)
    if db_path is None:
        raise ValueError(f"Could not resolve BIRD sqlite for db_id={db_id!r}. Set BIRD_SQLITE_ROOT.")
    df = _execute_sqlite_query_to_df(db_path, bird_sql_for_sqlite(sql), max_rows=max_rows)
    if df.empty:
        raise ValueError("Empty result")

    n_rows, n_cols = int(len(df)), int(len(df.columns))
    table = pa.Table.from_pandas(df, preserve_index=False)
    comp = _get_default_compression()
    enc_strat = _get_default_encoding_strategy()
    pq_bytes = _encode_parquet(table, comp, enc_strat)

    results_dir = SCRATCH / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    mid = str(uuid.uuid4())
    path = results_dir / "materialized" / f"bird_exec_{mid}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pq_bytes)

    cfg = ResultConfig(
        n_rows=n_rows,
        n_cols=n_cols,
        compression=comp,
        encoding_strategy=enc_strat,
        materialized_path=path,
    )
    # F7 + F10: pre-populate hints and payload caches so describe / large_json
    # / large_result_auto / blob endpoints all skip the Parquet re-read. We
    # already have `table` and `pq_bytes` at the default codec; reuse them.
    try:
        _populate_materialized_caches(
            cfg,
            result_id=mid,
            df=df,
            table=table,
            parquet_bytes=pq_bytes,
        )
    except Exception:
        cfg.cached_hints = None

    RESULT_REGISTRY[mid] = cfg
    return {"result_id": mid, "n_rows": n_rows, "n_cols": n_cols}


@mcp.tool()
def bird_query_auto(
    db_id: str,
    sql: str,
    optimization_target: Optional[str] = None,
    rows_per_chunk: int = 8192,
    prefer_streaming: bool = False,
    use_mab: bool = False,
    max_rows: int = 500_000,
) -> CallToolResult:
    """
    Execute BIRD SQL on the server and return the payload using one-shot server-side selection.

    Internally materializes once and delegates to large_result_auto so the response
    shape matches docs/server_side_format_selection.md.
    """
    mat = bird_query_materialize(db_id=db_id, sql=sql, max_rows=max_rows)
    rid = str(mat.get("result_id") or "")
    nr = int(mat.get("n_rows") or 0)
    nc = int(mat.get("n_cols") or 0)
    if not rid:
        raise RuntimeError("bird_query_materialize returned empty result_id")
    return large_result_auto(
        n_rows=nr,
        n_cols=nc,
        rows_per_chunk=rows_per_chunk,
        result_id=rid,
        optimization_target=optimization_target,
        prefer_streaming=prefer_streaming,
        use_mab=use_mab,
    )


@mcp.tool()
def bird_query_run_inline(
    db_id: str,
    sql: str,
    optimization_target: Optional[str] = None,
    rows_per_chunk: int = 8192,
    prefer_streaming: bool = False,
    use_mab: bool = False,
    max_rows: int = 500_000,
) -> CallToolResult:
    """
    Round-2 (F9): execute BIRD SQL + select format + return payload in a *single*
    MCP round trip, avoiding the wasted Parquet disk write when JSON wins.

    Workflow:

    1. Execute SQL on the SQLite db.
    2. Build size hints in memory (no disk).
    3. Run the same `select_format_with_hints` selector as `large_result_auto`.
    4. If chosen format is JSON: return inline records (no parquet, no
       result_id allocated, no disk I/O). This is the BIRD common case.
    5. Otherwise: persist parquet, register a ResultConfig with caches
       primed, and return a descriptor whose shape matches `large_result_auto`.

    Compared to `bird_query_auto` this saves one parquet encode + one disk
    write on the JSON path (which is ~99% of BIRD queries).
    """
    db_path = _resolve_bird_sqlite_path(db_id)
    if db_path is None:
        raise ValueError(
            f"Could not resolve BIRD sqlite for db_id={db_id!r}. "
            f"Set BIRD_SQLITE_ROOT."
        )
    df = _execute_sqlite_query_to_df(
        db_path, bird_sql_for_sqlite(sql), max_rows=max_rows,
    )
    if df.empty:
        raise ValueError("Empty result")

    n_rows = int(len(df))
    n_cols = int(len(df.columns))

    # Hints in memory — same logic as `_compute_tabular_size_hints_from_df`,
    # but keyed off the live DataFrame (no Parquet round trip).
    comp = _get_default_compression()
    enc_strat = _get_default_encoding_strategy()
    ipc_comp = _get_default_arrow_ipc_compression()
    table = pa.Table.from_pandas(df, preserve_index=False)
    hints = _compute_tabular_size_hints_from_df(
        df,
        rows_per_chunk=rows_per_chunk,
        comp=comp,
        enc_strat=enc_strat,
        ipc_comp=ipc_comp,
        table=table,
    )
    resolved_rows = int(hints.get("resolved_n_rows") or n_rows)
    resolved_cols = int(hints.get("resolved_n_cols") or n_cols)

    if optimization_target:
        try:
            sel_target = OptimizationTarget(optimization_target.strip().lower())
        except ValueError:
            sel_target = get_default_target()
    else:
        sel_target = get_default_target()

    sel_ctx = SelectionContext(
        n_rows=resolved_rows,
        n_cols=resolved_cols,
        target=sel_target,
        prefer_streaming=prefer_streaming,
    )
    hints_for_select: Dict[str, Any] = {
        "json_bytes": int(hints["json_bytes"]),
        "parquet_bytes": int(hints["parquet_bytes"]),
        "parquet_stream_first_chunk_bytes": int(hints["parquet_stream_first_chunk_bytes"]),
        "arrow_ipc_bytes": int(hints["arrow_ipc_bytes"]),
        "arrow_ipc_stream_first_chunk_bytes": int(hints["arrow_ipc_stream_first_chunk_bytes"]),
    }
    mab_state = load_mab_state(_SERVER_MAB_STATE_PATH) if use_mab else None
    chosen = (
        select_format_with_mab(sel_ctx, hints_for_select, mab_state)
        if use_mab
        else select_format_with_hints(sel_ctx, hints_for_select)
    )

    # Fast path: inline JSON without a parquet disk write or result_id alloc.
    if chosen == "json":
        try:
            cap = _json_cells_cap()
            cells = resolved_rows * resolved_cols
            if cap is not None and cells > cap:
                raise ValueError(
                    f"Result too large for JSON ({cells} cells > {cap})"
                )
            records = df.to_dict(orient="records")
        except Exception:
            chosen = "parquet_blob"
        else:
            structured = {
                "payload_kind": "tabular",
                "chosen_format": "json",
                "optimization_target": sel_target.value,
                "payload": {"kind": "json", "records": records},
                "decode": {"encoding": "json_records", "transport": "inline"},
            }
            return CallToolResult(
                content=[TextContent(type="text", text="Returning JSON records (json).")],
                structured_content=structured,
            )

    # Slow path: chosen format needs a backing result_id + disk file. Persist
    # parquet now, prime caches, and delegate to large_result_auto's
    # descriptor branches via the `result_id` we just allocated.
    pq_bytes = _encode_parquet(table, comp, enc_strat)
    results_dir = SCRATCH / "results"
    mid = str(uuid.uuid4())
    path = results_dir / "materialized" / f"bird_inline_{mid}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pq_bytes)

    cfg = ResultConfig(
        n_rows=n_rows,
        n_cols=n_cols,
        compression=comp,
        encoding_strategy=enc_strat,
        materialized_path=path,
    )
    try:
        _populate_materialized_caches(
            cfg,
            result_id=mid,
            df=df,
            table=table,
            parquet_bytes=pq_bytes,
        )
    except Exception:
        cfg.cached_hints = None
    RESULT_REGISTRY[mid] = cfg

    return large_result_auto(
        n_rows=resolved_rows,
        n_cols=resolved_cols,
        rows_per_chunk=rows_per_chunk,
        result_id=mid,
        optimization_target=optimization_target,
        prefer_streaming=prefer_streaming,
        use_mab=use_mab,
    )


@mcp.tool()
def large_json(
    n_rows: int,
    n_cols: int,
    result_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Return the full dataset as JSON (baseline).

    When *result_id* refers to a previously registered materialized result
    (via POST /materialized), the actual data from that file is returned
    and n_rows/n_cols are informational only.
    """
    if result_id is not None:
        config = RESULT_REGISTRY.get(result_id)
        if config is None:
            raise ValueError(f"Unknown result_id: {result_id}")
        # F7: serve the pre-built records list when present (BIRD hot path).
        if config.cached_json_records is not None:
            _payload_cache_touch(result_id)
            return config.cached_json_records
        df = _resolve_dataframe(config)
        cells = len(df) * len(df.columns)
        cap = _json_cells_cap()
        if cap is not None and cells > cap:
            raise ValueError(
                f"Result too large for JSON ({cells} cells > {cap})"
            )
        return df.to_dict(orient="records")

    df = _generate_dataframe(n_rows=n_rows, n_cols=n_cols)
    return df.to_dict(orient="records")


@mcp.tool()
def large_parquet_blob(
    n_rows: int,
    n_cols: int,
    compression: Optional[str] = None,
    encoding_strategy: Optional[str] = None,
    result_id: Optional[str] = None,
) -> CallToolResult:
    """
    Prepare a Parquet blob and return a descriptor with a download URL.

    When *result_id* refers to a materialized result the blob is encoded
    from that data.  A new internal ID is allocated for the HTTP endpoint.
    """
    comp = compression or _get_default_compression()
    enc_strat = encoding_strategy or _get_default_encoding_strategy()

    materialized_path: Path | None = None
    if result_id is not None:
        src = RESULT_REGISTRY.get(result_id)
        if src is None:
            raise ValueError(f"Unknown result_id: {result_id}")
        n_rows = src.n_rows
        n_cols = src.n_cols
        materialized_path = src.materialized_path

    new_id = str(uuid.uuid4())
    RESULT_REGISTRY[new_id] = ResultConfig(
        n_rows=n_rows,
        n_cols=n_cols,
        compression=comp,
        encoding_strategy=enc_strat,
        materialized_path=materialized_path,
        arm="parquet_blob",
    )
    _inherit_caches(new_id, result_id)

    descriptor = {
        "mode": "parquet_blob",
        "id": new_id,
        "url": f"http://localhost:8000/blobs/{new_id}.parquet",
        "n_rows": n_rows,
        "n_cols": n_cols,
        "compression": comp,
        "encoding_strategy": enc_strat,
    }
    _add_arm_fields(descriptor, "parquet_blob")

    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=f"Parquet blob prepared with id={new_id}, "
                f"rows={n_rows}, cols={n_cols}, "
                f"compression={comp}, encoding_strategy={enc_strat}",
            )
        ],
        structured_content=descriptor,
        _meta={"result_id": new_id},
    )


@mcp.tool()
def large_parquet_stream(
    n_rows: int,
    n_cols: int,
    rows_per_chunk: int,
    compression: Optional[str] = None,
    encoding_strategy: Optional[str] = None,
    result_id: Optional[str] = None,
) -> CallToolResult:
    """
    Prepare a Parquet streaming endpoint and return a descriptor.

    When *result_id* refers to a materialized result the stream chunks
    are drawn from that data.
    """
    if rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be positive")

    comp = compression or _get_default_compression()
    enc_strat = encoding_strategy or _get_default_encoding_strategy()

    materialized_path: Path | None = None
    if result_id is not None:
        src = RESULT_REGISTRY.get(result_id)
        if src is None:
            raise ValueError(f"Unknown result_id: {result_id}")
        n_rows = src.n_rows
        n_cols = src.n_cols
        materialized_path = src.materialized_path

    new_id = str(uuid.uuid4())
    RESULT_REGISTRY[new_id] = ResultConfig(
        n_rows=n_rows,
        n_cols=n_cols,
        rows_per_chunk=rows_per_chunk,
        compression=comp,
        encoding_strategy=enc_strat,
        materialized_path=materialized_path,
        arm="parquet_stream",
    )
    _inherit_caches(new_id, result_id)

    descriptor = {
        "mode": "parquet_stream",
        "id": new_id,
        "url": f"http://localhost:8000/streams/{new_id}",
        "n_rows": n_rows,
        "n_cols": n_cols,
        "rows_per_chunk": rows_per_chunk,
        "compression": comp,
        "encoding_strategy": enc_strat,
    }
    _add_arm_fields(descriptor, "parquet_stream")

    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=(
                    "Parquet stream prepared with "
                    f"id={new_id}, rows={n_rows}, cols={n_cols}, "
                    f"rows_per_chunk={rows_per_chunk}, "
                    f"compression={comp}, encoding_strategy={enc_strat}"
                ),
            )
        ],
        structured_content=descriptor,
        _meta={"result_id": new_id},
    )


@mcp.tool()
def large_arrow_ipc_blob(
    n_rows: int,
    n_cols: int,
    ipc_compression: Optional[str] = None,
    result_id: Optional[str] = None,
) -> CallToolResult:
    """
    Prepare an Arrow IPC file blob and return a descriptor with a download URL.

    When *result_id* refers to a materialized result the IPC file is encoded
    from that data. A new internal ID is allocated for the HTTP endpoint.
    """
    ipc_comp = ipc_compression or _get_default_arrow_ipc_compression()
    if ipc_comp not in VALID_ARROW_IPC_COMPRESSIONS:
        ipc_comp = "none"

    materialized_path: Path | None = None
    if result_id is not None:
        src = RESULT_REGISTRY.get(result_id)
        if src is None:
            raise ValueError(f"Unknown result_id: {result_id}")
        n_rows = src.n_rows
        n_cols = src.n_cols
        materialized_path = src.materialized_path

    new_id = str(uuid.uuid4())
    RESULT_REGISTRY[new_id] = ResultConfig(
        n_rows=n_rows,
        n_cols=n_cols,
        ipc_compression=ipc_comp,
        materialized_path=materialized_path,
        arm="arrow_ipc_blob",
    )
    _inherit_caches(new_id, result_id)

    descriptor = {
        "mode": "arrow_ipc_blob",
        "id": new_id,
        "url": f"http://localhost:8000/ipc-blobs/{new_id}.arrow",
        "n_rows": n_rows,
        "n_cols": n_cols,
        "ipc_compression": ipc_comp,
    }
    _add_arm_fields(descriptor, "arrow_ipc_blob")

    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=(
                    f"Arrow IPC blob prepared with id={new_id}, "
                    f"rows={n_rows}, cols={n_cols}, ipc_compression={ipc_comp}"
                ),
            )
        ],
        structured_content=descriptor,
        _meta={"result_id": new_id},
    )


@mcp.tool()
def large_arrow_ipc_stream(
    n_rows: int,
    n_cols: int,
    rows_per_chunk: int,
    ipc_compression: Optional[str] = None,
    result_id: Optional[str] = None,
) -> CallToolResult:
    """
    Prepare an Arrow IPC streaming endpoint (length-prefixed IPC file chunks).

    When *result_id* refers to a materialized result the stream chunks
    are drawn from that data.
    """
    if rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be positive")

    ipc_comp = ipc_compression or _get_default_arrow_ipc_compression()
    if ipc_comp not in VALID_ARROW_IPC_COMPRESSIONS:
        ipc_comp = "none"

    materialized_path: Path | None = None
    if result_id is not None:
        src = RESULT_REGISTRY.get(result_id)
        if src is None:
            raise ValueError(f"Unknown result_id: {result_id}")
        n_rows = src.n_rows
        n_cols = src.n_cols
        materialized_path = src.materialized_path

    new_id = str(uuid.uuid4())
    RESULT_REGISTRY[new_id] = ResultConfig(
        n_rows=n_rows,
        n_cols=n_cols,
        rows_per_chunk=rows_per_chunk,
        ipc_compression=ipc_comp,
        materialized_path=materialized_path,
        arm="arrow_ipc_stream",
    )
    _inherit_caches(new_id, result_id)

    descriptor = {
        "mode": "arrow_ipc_stream",
        "id": new_id,
        "url": f"http://localhost:8000/ipc-streams/{new_id}",
        "n_rows": n_rows,
        "n_cols": n_cols,
        "rows_per_chunk": rows_per_chunk,
        "ipc_compression": ipc_comp,
    }
    _add_arm_fields(descriptor, "arrow_ipc_stream")

    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=(
                    "Arrow IPC stream prepared with "
                    f"id={new_id}, rows={n_rows}, cols={n_cols}, "
                    f"rows_per_chunk={rows_per_chunk}, ipc_compression={ipc_comp}"
                ),
            )
        ],
        structured_content=descriptor,
        _meta={"result_id": new_id},
    )


@mcp.tool()
def describe_result_formats(
    n_rows: int,
    n_cols: int,
    rows_per_chunk: int = 8192,
    result_id: Optional[str] = None,
    optimization_target: Optional[str] = None,
    prefer_streaming: bool = False,
) -> CallToolResult:
    """
    Return format hints (approx bytes per format) for Workflow A (client-driven
    format selection).  When *result_id* refers to a materialized result the
    hints reflect the real data.

    Optional *optimization_target* (min_bytes | min_latency | min_time_to_first_rows)
    and *prefer_streaming* are used with the same logic as format_selector
    to populate *recommended_format* (server-side mirror of client selection).
    If *optimization_target* is omitted, FORMAT_SELECT_TARGET on the server applies.
    """
    # If result_id points at unstructured data, return unstructured hints.
    if result_id is not None:
        cfg = RESULT_REGISTRY.get(result_id)
        if cfg is not None and cfg.payload_kind == "unstructured":
            if cfg.raw_path is None or not cfg.raw_path.is_file():
                raise ValueError(f"Unstructured result_id missing file: {result_id}")
            unstructured = _unstructured_hints(cfg, result_id)
            raw_bytes = unstructured["raw_bytes"]
            gzip_bytes = unstructured["gzip_bytes"]
            inline_bytes = unstructured["text_inline_bytes"]
            if optimization_target:
                try:
                    sel_target = OptimizationTarget(optimization_target.strip().lower())
                except ValueError:
                    sel_target = get_default_target()
            else:
                sel_target = get_default_target()
            hints_for_select: Dict[str, Any] = unstructured
            sel_ctx = SelectionContext(n_rows=0, n_cols=0, target=sel_target, prefer_streaming=prefer_streaming)
            recommended = select_format_with_hints(sel_ctx, hints_for_select)
            structured = {
                "payload_kind": "unstructured",
                "mime_type": cfg.raw_mime_type or "application/octet-stream",
                "recommended_format": recommended,
                "recommendation_target": sel_target.value,
                "formats": {
                    "raw_blob": {"supported": True, "approx_bytes": int(raw_bytes)},
                    "gzip_blob": {"supported": True, "approx_bytes": int(gzip_bytes) if gzip_bytes is not None else None},
                    "zstd_blob": {"supported": True, "approx_bytes": unstructured["zstd_bytes"]},
                    "brotli_blob": {"supported": True, "approx_bytes": unstructured["brotli_bytes"]},
                    "text_inline": {"supported": inline_bytes is not None, "approx_bytes": int(inline_bytes) if inline_bytes is not None else None},
                },
            }
            return CallToolResult(
                content=[TextContent(type="text", text=f"Unstructured hints: raw={raw_bytes} B, gzip={gzip_bytes} B, zstd={unstructured['zstd_bytes']} B, brotli={unstructured['brotli_bytes']} B; recommended_format={recommended}")],
                structured_content=structured,
            )

    comp = _get_default_compression()
    enc_strat = _get_default_encoding_strategy()
    ipc_comp = _get_default_arrow_ipc_compression()

    hints = _get_tabular_size_hints_cached(
        n_rows=n_rows,
        n_cols=n_cols,
        rows_per_chunk=rows_per_chunk,
        result_id=result_id,
        comp=comp,
        enc_strat=enc_strat,
        ipc_comp=ipc_comp,
    )
    n_rows = int(hints.get("resolved_n_rows") or n_rows)
    n_cols = int(hints.get("resolved_n_cols") or n_cols)
    json_bytes = int(hints["json_bytes"])
    parquet_bytes = int(hints["parquet_bytes"])
    parquet_stream_first_chunk_bytes = int(hints["parquet_stream_first_chunk_bytes"])
    arrow_ipc_bytes = int(hints["arrow_ipc_bytes"])
    arrow_ipc_stream_first_chunk_bytes = int(hints["arrow_ipc_stream_first_chunk_bytes"])

    if optimization_target:
        try:
            sel_target = OptimizationTarget(optimization_target.strip().lower())
        except ValueError:
            sel_target = get_default_target()
    else:
        sel_target = get_default_target()

    hints_for_select: Dict[str, Any] = {
        "json_bytes": json_bytes,
        "parquet_bytes": parquet_bytes,
        "parquet_stream_first_chunk_bytes": parquet_stream_first_chunk_bytes,
        "arrow_ipc_bytes": arrow_ipc_bytes,
        "arrow_ipc_stream_first_chunk_bytes": arrow_ipc_stream_first_chunk_bytes,
    }
    sel_ctx = SelectionContext(
        n_rows=n_rows,
        n_cols=n_cols,
        target=sel_target,
        prefer_streaming=prefer_streaming,
    )
    recommended_format = select_format_with_hints(sel_ctx, hints_for_select)

    structured = {
        "approx_rows": n_rows,
        "approx_cols": n_cols,
        "parquet_compression": comp,
        "parquet_encoding_strategy": enc_strat,
        "arrow_ipc_compression": ipc_comp,
        "recommended_format": recommended_format,
        "recommendation_target": sel_target.value,
        "recommendation_prefer_streaming": prefer_streaming,
        "formats": {
            "json": {"supported": True, "approx_bytes": json_bytes},
            "parquet_blob": {"supported": True, "approx_bytes": parquet_bytes},
            "parquet_stream": {
                "supported": True,
                "approx_bytes": parquet_bytes,
                "approx_first_chunk_bytes": parquet_stream_first_chunk_bytes,
            },
            "arrow_ipc_blob": {
                "supported": True,
                "approx_bytes": arrow_ipc_bytes,
            },
            "arrow_ipc_stream": {
                "supported": True,
                "approx_bytes": arrow_ipc_bytes,
                "approx_first_chunk_bytes": arrow_ipc_stream_first_chunk_bytes,
            },
        },
    }
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=(
                    f"Format hints: json={json_bytes} B, parquet_blob={parquet_bytes} B, "
                    f"parquet stream first_chunk={parquet_stream_first_chunk_bytes} B, "
                    f"arrow_ipc_blob={arrow_ipc_bytes} B, "
                    f"arrow_ipc stream first_chunk={arrow_ipc_stream_first_chunk_bytes} B "
                    f"(parquet codec={comp}, encoding={enc_strat}, ipc={ipc_comp}); "
                    f"recommended_format={recommended_format} "
                    f"(target={sel_target.value}, prefer_streaming={prefer_streaming})"
                ),
            )
        ],
        structured_content=structured,
    )


@mcp.tool()
def describe_and_fetch_result(
    n_rows: int,
    n_cols: int,
    rows_per_chunk: int = 8192,
    result_id: Optional[str] = None,
    optimization_target: Optional[str] = None,
    prefer_streaming: bool = False,
    use_mab: bool = False,
) -> CallToolResult:
    """
    Fused enhanced workflow: format hints + heuristic selection + payload in one MCP call
    (same semantics as large_result_auto; eliminates the separate describe round trip).
    """
    return large_result_auto(
        n_rows=n_rows,
        n_cols=n_cols,
        rows_per_chunk=rows_per_chunk,
        result_id=result_id,
        optimization_target=optimization_target,
        prefer_streaming=prefer_streaming,
        use_mab=use_mab,
    )



# ----- Inline binary transport (experimental) -------------------------------------
# A blob arm normally costs the tool call plus a second request to fetch the bytes. For a small payload the bytes can ride
# in the tool result instead, in `_meta` (not in the content a host shows the model), saving the descriptor round trip
# and a connection. Off unless FORMAT_INLINE_MAX_BYTES > 0.
INLINE_META_KEY = "transform/inline"


def _inline_limit_bytes() -> int:
    try:
        return max(0, int(os.environ.get("FORMAT_INLINE_MAX_BYTES", "0").strip() or 0))
    except ValueError:
        return 0


def _should_inline(encoded_bytes: int) -> bool:
    limit = _inline_limit_bytes()
    if limit <= 0 or encoded_bytes > limit:
        return False
    try:
        rtt_ms = float(os.environ.get("FORMAT_LATENCY_RTT_MS", "0").strip() or 0)
    except ValueError:
        rtt_ms = 0.0
    mbps = _get_network_mbps()
    if rtt_ms > 0 and mbps > 0:
        # base64 adds a third; inline while that extra transfer costs less than the round trip it saves
        return 0.34 * encoded_bytes * 8 / (mbps * 1e6) < rtt_ms / 1000
    return True


def _maybe_inline(result: CallToolResult, get_data, size_hint: Optional[int], mime: str) -> CallToolResult:
    """Return *result* with the payload bytes attached in `_meta` when the arm is small enough, else unchanged."""
    if size_hint is not None and not _should_inline(int(size_hint)):
        return result
    if size_hint is None and _inline_limit_bytes() <= 0:
        return result
    data = get_data()
    if data is None or not _should_inline(len(data)):
        return result
    structured = dict(result.structured_content or {})
    structured["decode"] = {**structured.get("decode", {}), "transport": "inline_meta", "meta_key": INLINE_META_KEY,
                            "bytes": len(data)}
    meta = {INLINE_META_KEY: {"encoding": "base64", "mime_type": mime, "bytes": len(data),
                              "sha256": hashlib.sha256(data).hexdigest(), "data": base64.b64encode(data).decode("ascii")}}
    return CallToolResult(content=result.content, structured_content=structured, _meta=meta)




def _table_arm_candidates(config: ResultConfig, container: str):
    """(arm, encode, decode, already_encoded) for the container's default and its registered codec arms."""
    strategy = config.encoding_strategy or _get_default_encoding_strategy()
    if container == "parquet":
        default = ("parquet_blob", lambda t: _encode_parquet(t, config.compression or _get_default_compression(), strategy),
                   lambda b: pq.read_table(BytesIO(b)), config.cached_parquet_blob_bytes is not None)
        others = [(a.name, (lambda t, a=a: arms.encode_parquet(t, a.codec, a.level, strategy)), lambda b: pq.read_table(BytesIO(b)), False)
                  for a in REGISTRY.for_modality("table") if a.container == "parquet" and a.transport == "blob" and not a.selectable]
    else:
        default = ("arrow_ipc_blob", lambda t: _encode_arrow_ipc_file(t, config.ipc_compression or _get_default_arrow_ipc_compression()),
                   lambda b: pa.ipc.open_file(pa.BufferReader(b)).read_all(), config.cached_arrow_ipc_blob_bytes is not None)
        others = [(a.name, (lambda t, a=a: arms.encode_arrow_ipc(t, a.codec)), lambda b: pa.ipc.open_file(pa.BufferReader(b)).read_all(), False)
                  for a in REGISTRY.for_modality("table") if a.container == "arrow_ipc" and a.transport == "blob" and not a.selectable]
    return [default, *others]


def _cost_aware_table_arm(result_id: Optional[str], chosen: str, default_bytes: int, target: OptimizationTarget) -> Optional[str]:
    """Registered table codec arm that beats the default for this result on this link, or None to keep the default.

    Only with FORMAT_COST_AWARE=1, the latency target, a known link speed and a registered table; a default that crosses
    the link in a few milliseconds is kept without trying anything.
    """
    mbps = _get_network_mbps()
    if not (cost_model.enabled() and target == OptimizationTarget.MIN_LATENCY and mbps > 0 and result_id):
        return None
    cfg = RESULT_REGISTRY.get(result_id)
    if cfg is None or cfg.payload_kind != "tabular" or cfg.materialized_path is None and cfg.cached_arrow_table is None:
        return None
    if default_bytes * 8.0 / (mbps * 1e6) <= cost_model.TABLE_SKIP_TRANSFER_S:
        return None
    table = _resolve_arrow_table(cfg)
    sample = table.slice(0, cost_model.TABLE_SAMPLE_ROWS)
    container = "parquet" if chosen == "parquet_blob" else "arrow_ipc"
    best, _ = cost_model.choose_table_arm(sample, table.num_rows, _table_arm_candidates(cfg, container), mbps)
    return None if best == chosen else best


@mcp.tool()
def large_result_auto(
    n_rows: int,
    n_cols: int,
    rows_per_chunk: int = 8192,
    result_id: Optional[str] = None,
    optimization_target: Optional[str] = None,
    prefer_streaming: bool = False,
    use_mab: bool = False,
) -> CallToolResult:
    """
    One-shot server-side format selection: choose the best representation and return it.

    This is the "one MCP round trip" path: the client sends (result_id or shape) and
    optimization_target; the server returns payload + chosen_format + decode hints.
    """
    # Unstructured branch (result_id required).
    if result_id is not None:
        cfg = RESULT_REGISTRY.get(result_id)
        if cfg is not None and cfg.payload_kind == "unstructured":
            if cfg.raw_path is None or not cfg.raw_path.is_file():
                raise ValueError(f"Unstructured result_id missing file: {result_id}")
            if optimization_target:
                try:
                    sel_target = OptimizationTarget(optimization_target.strip().lower())
                except ValueError:
                    sel_target = get_default_target()
            else:
                sel_target = get_default_target()
            # Sizes are estimated (not computed exactly) only when the target is latency, the link speed is known and
            # no bandit is learning from exact sizes; min_bytes keeps exact sizes because it asks for the smallest payload.
            cost_aware = cost_model.enabled() and sel_target == OptimizationTarget.MIN_LATENCY and not use_mab
            unstructured = _unstructured_hints(cfg, result_id, cost_aware_mbps=_get_network_mbps() if cost_aware else 0.0)
            sel_ctx = SelectionContext(n_rows=0, n_cols=0, target=sel_target, prefer_streaming=prefer_streaming)
            hints: Dict[str, Any] = unstructured
            mab_state = load_mab_state(_SERVER_MAB_STATE_PATH) if use_mab else None
            chosen = select_format_with_mab(sel_ctx, hints, mab_state) if use_mab else select_format_with_hints(sel_ctx, hints)

            if chosen == "text_inline":
                data = cfg.raw_path.read_bytes()
                # Best-effort decode; if it fails, fall back to blob.
                try:
                    text = data.decode(cfg.raw_charset or "utf-8")
                except Exception:
                    chosen = "raw_blob"
                else:
                    structured = {
                        "payload_kind": "unstructured",
                        "chosen_format": "text_inline",
                        "optimization_target": sel_target.value,
                        "payload": {"kind": "text", "text": text},
                        "decode": {
                            "encoding": "text",
                            "transport": "inline",
                            "mime_type": cfg.raw_mime_type or "text/plain",
                            "charset": cfg.raw_charset or "utf-8",
                        },
                    }
                    return CallToolResult(
                        content=[TextContent(type="text", text="Returning inline text payload (text_inline).")],
                        structured_content=structured,
                    )

            if chosen in RAW_BYTE_CODECS:
                codec, _, route, _ = RAW_BYTE_CODECS[chosen]
                new_id = str(uuid.uuid4())
                RESULT_REGISTRY[new_id] = ResultConfig(
                    n_rows=0,
                    n_cols=0,
                    payload_kind="unstructured",
                    raw_path=cfg.raw_path,
                    raw_mime_type=cfg.raw_mime_type,
                    raw_charset=cfg.raw_charset,
                    raw_gzip_path=cfg.raw_gzip_path,
                    raw_codec_paths=cfg.raw_codec_paths,
                    arm=chosen,
                )
                url = f"http://localhost:8000/{route}/{new_id}"
                mime = cfg.raw_mime_type or "application/octet-stream"
                structured = {
                    "payload_kind": "unstructured",
                    "chosen_format": chosen,
                    "optimization_target": sel_target.value,
                    "payload": _add_arm_fields({"kind": "descriptor", "mode": chosen, "id": new_id, "url": url, "mime_type": mime}, chosen),
                    "decode": {
                        "encoding": "raw_bytes",
                        "transport": "http_blob",
                        "url": url,
                        "mime_type": mime,
                        "codec": codec,
                        "uncompressed_bytes": int(unstructured["raw_bytes"]),
                    },
                }
                return _maybe_inline(
                    CallToolResult(
                        content=[TextContent(type="text", text=f"Returning {codec} blob descriptor ({chosen}): {url}")],
                        structured_content=structured,
                    ),
                    lambda: _raw_codec_blob(cfg, result_id, chosen), None, mime,
                )

            if chosen == "gzip_blob":
                new_id = str(uuid.uuid4())
                RESULT_REGISTRY[new_id] = ResultConfig(
                    n_rows=0,
                    n_cols=0,
                    payload_kind="unstructured",
                    raw_path=cfg.raw_path,
                    raw_mime_type=cfg.raw_mime_type,
                    raw_charset=cfg.raw_charset,
                    raw_gzip_path=cfg.raw_gzip_path,
                    arm="gzip_blob",
                )
                descriptor = {
                    "mode": "gzip_blob",
                    "id": new_id,
                    "url": f"http://localhost:8000/raw-gzip/{new_id}",
                    "mime_type": cfg.raw_mime_type or "application/octet-stream",
                }
                structured = {
                    "payload_kind": "unstructured",
                    "chosen_format": "gzip_blob",
                    "optimization_target": sel_target.value,
                    "payload": _add_arm_fields({"kind": "descriptor", **descriptor}, "gzip_blob"),
                    "decode": {
                        "encoding": "raw_bytes",
                        "transport": "http_blob",
                        "url": descriptor["url"],
                        "mime_type": descriptor["mime_type"],
                        "content_encoding": "gzip",
                    },
                }
                return _maybe_inline(
                    CallToolResult(
                        content=[TextContent(type="text", text=f"Returning gzip blob descriptor (gzip_blob): {descriptor['url']}")],
                        structured_content=structured,
                    ),
                    lambda: _raw_gzip_blob(cfg, result_id), None, descriptor["mime_type"],
                )

            # raw_blob default
            new_id = str(uuid.uuid4())
            RESULT_REGISTRY[new_id] = ResultConfig(
                n_rows=0,
                n_cols=0,
                payload_kind="unstructured",
                raw_path=cfg.raw_path,
                raw_mime_type=cfg.raw_mime_type,
                raw_charset=cfg.raw_charset,
                raw_gzip_path=cfg.raw_gzip_path,
                arm="raw_blob",
            )
            descriptor = {
                "mode": "raw_blob",
                "id": new_id,
                "url": f"http://localhost:8000/raw/{new_id}",
                "mime_type": cfg.raw_mime_type or "application/octet-stream",
            }
            structured = {
                "payload_kind": "unstructured",
                "chosen_format": "raw_blob",
                "optimization_target": sel_target.value,
                "payload": _add_arm_fields({"kind": "descriptor", **descriptor}, "raw_blob"),
                "decode": {
                    "encoding": "raw_bytes",
                    "transport": "http_blob",
                    "url": descriptor["url"],
                    "mime_type": descriptor["mime_type"],
                },
            }
            return _maybe_inline(
                CallToolResult(
                    content=[TextContent(type="text", text=f"Returning raw blob descriptor (raw_blob): {descriptor['url']}")],
                    structured_content=structured,
                ),
                lambda: cfg.raw_path.read_bytes(), int(unstructured["raw_bytes"]), descriptor["mime_type"],
            )

    # Tabular branch.
    comp = _get_default_compression()
    enc_strat = _get_default_encoding_strategy()
    ipc_comp = _get_default_arrow_ipc_compression()

    hints = _get_tabular_size_hints_cached(
        n_rows=n_rows,
        n_cols=n_cols,
        rows_per_chunk=rows_per_chunk,
        result_id=result_id,
        comp=comp,
        enc_strat=enc_strat,
        ipc_comp=ipc_comp,
    )
    resolved_rows = int(hints.get("resolved_n_rows") or n_rows)
    resolved_cols = int(hints.get("resolved_n_cols") or n_cols)

    if optimization_target:
        try:
            sel_target = OptimizationTarget(optimization_target.strip().lower())
        except ValueError:
            sel_target = get_default_target()
    else:
        sel_target = get_default_target()

    sel_ctx = SelectionContext(
        n_rows=resolved_rows,
        n_cols=resolved_cols,
        target=sel_target,
        prefer_streaming=prefer_streaming,
    )
    hints_for_select: Dict[str, Any] = {
        "json_bytes": int(hints["json_bytes"]),
        "parquet_bytes": int(hints["parquet_bytes"]),
        "parquet_stream_first_chunk_bytes": int(hints["parquet_stream_first_chunk_bytes"]),
        "arrow_ipc_bytes": int(hints["arrow_ipc_bytes"]),
        "arrow_ipc_stream_first_chunk_bytes": int(hints["arrow_ipc_stream_first_chunk_bytes"]),
    }

    mab_state = load_mab_state(_SERVER_MAB_STATE_PATH) if use_mab else None
    chosen = select_format_with_mab(sel_ctx, hints_for_select, mab_state) if use_mab else select_format_with_hints(sel_ctx, hints_for_select)

    if chosen == "json":
        try:
            records = large_json(resolved_rows, resolved_cols, result_id=result_id)
        except Exception:
            chosen = "parquet_blob"
        else:
            structured = {
                "payload_kind": "tabular",
                "chosen_format": "json",
                "optimization_target": sel_target.value,
                "payload": {"kind": "json", "records": records},
                "decode": {"encoding": "json_records", "transport": "inline"},
            }
            return CallToolResult(
                content=[TextContent(type="text", text="Returning JSON records (json).")],
                structured_content=structured,
            )

    if chosen == "parquet_stream":
        desc = large_parquet_stream(
            resolved_rows,
            resolved_cols,
            rows_per_chunk=rows_per_chunk,
            compression=comp,
            encoding_strategy=enc_strat,
            result_id=result_id,
        ).structured_content or {}
        structured = {
            "payload_kind": "tabular",
            "chosen_format": "parquet_stream",
            "optimization_target": sel_target.value,
            "payload": {"kind": "descriptor", **desc},
            "decode": {
                "encoding": "parquet",
                "transport": "http_length_prefixed_stream",
                "url": desc.get("url"),
                "rows_per_chunk": desc.get("rows_per_chunk"),
            },
        }
        return CallToolResult(
            content=[TextContent(type="text", text=f"Returning Parquet stream descriptor (parquet_stream): {desc.get('url')}")],
            structured_content=structured,
        )

    if chosen == "arrow_ipc_stream":
        desc = large_arrow_ipc_stream(
            resolved_rows,
            resolved_cols,
            rows_per_chunk=rows_per_chunk,
            ipc_compression=ipc_comp,
            result_id=result_id,
        ).structured_content or {}
        structured = {
            "payload_kind": "tabular",
            "chosen_format": "arrow_ipc_stream",
            "optimization_target": sel_target.value,
            "payload": {"kind": "descriptor", **desc},
            "decode": {
                "encoding": "arrow_ipc",
                "transport": "http_length_prefixed_stream",
                "url": desc.get("url"),
                "rows_per_chunk": desc.get("rows_per_chunk"),
            },
        }
        return CallToolResult(
            content=[TextContent(type="text", text=f"Returning Arrow IPC stream descriptor (arrow_ipc_stream): {desc.get('url')}")],
            structured_content=structured,
        )

    alt_arm = _cost_aware_table_arm(result_id, chosen, int(hints_for_select["parquet_bytes"] if chosen == "parquet_blob"
                                                         else hints_for_select["arrow_ipc_bytes"]), sel_target) \
        if chosen in ("parquet_blob", "arrow_ipc_blob") else None
    if alt_arm is not None:
        spec = REGISTRY.get(alt_arm)
        desc = fetch_arm(alt_arm, result_id).structured_content or {}
        structured = {
            "payload_kind": "tabular",
            "chosen_format": chosen,
            "optimization_target": sel_target.value,
            "payload": {"kind": "descriptor", **desc},
            "decode": {"encoding": "parquet" if spec.container == "parquet" else "arrow_ipc", "transport": "http_blob",
                       "url": desc.get("url"), "arm": alt_arm, "codec": spec.codec, "level": spec.level},
        }
        return CallToolResult(
            content=[TextContent(type="text", text=f"Returning {alt_arm} descriptor (cost-aware codec choice for {chosen}): {desc.get('url')}")],
            structured_content=structured,
        )

    if chosen == "arrow_ipc_blob":
        desc = large_arrow_ipc_blob(
            resolved_rows,
            resolved_cols,
            ipc_compression=ipc_comp,
            result_id=result_id,
        ).structured_content or {}
        structured = {
            "payload_kind": "tabular",
            "chosen_format": "arrow_ipc_blob",
            "optimization_target": sel_target.value,
            "payload": {"kind": "descriptor", **desc},
            "decode": {"encoding": "arrow_ipc", "transport": "http_blob", "url": desc.get("url")},
        }
        return _maybe_inline(
            CallToolResult(
                content=[TextContent(type="text", text=f"Returning Arrow IPC blob descriptor (arrow_ipc_blob): {desc.get('url')}")],
                structured_content=structured,
            ),
            lambda: _ipc_blob_data(RESULT_REGISTRY[desc["id"]], desc["id"]),
            hints_for_select["arrow_ipc_bytes"], "application/vnd.apache.arrow.file",
        )

    # parquet_blob default
    desc = large_parquet_blob(
        resolved_rows,
        resolved_cols,
        compression=comp,
        encoding_strategy=enc_strat,
        result_id=result_id,
    ).structured_content or {}
    structured = {
        "payload_kind": "tabular",
        "chosen_format": "parquet_blob",
        "optimization_target": sel_target.value,
        "payload": {"kind": "descriptor", **desc},
        "decode": {"encoding": "parquet", "transport": "http_blob", "url": desc.get("url")},
    }
    return _maybe_inline(
        CallToolResult(
            content=[TextContent(type="text", text=f"Returning Parquet blob descriptor (parquet_blob): {desc.get('url')}")],
            structured_content=structured,
        ),
        lambda: _parquet_blob_data(RESULT_REGISTRY[desc["id"]], desc["id"]),
        hints_for_select["parquet_bytes"], "application/octet-stream",
    )


@mcp.tool()
def record_format_outcome(
    n_rows: int,
    n_cols: int,
    optimization_target: str,
    format_used: str,
    *,
    bytes: Optional[int] = None,
    latency_s: Optional[float] = None,
    time_to_first_rows_s: Optional[float] = None,
) -> CallToolResult:
    """
    Update server-side MAB state with an observed outcome.

    This keeps the "one MCP round trip" property of large_result_auto intact:
    the outcome reporting is optional and typically used by benchmarking.
    """
    try:
        target = OptimizationTarget(optimization_target.strip().lower())
    except ValueError:
        target = get_default_target()

    ctx = SelectionContext(n_rows=n_rows, n_cols=n_cols, target=target)
    outcome: Dict[str, Any] = {
        "bytes": bytes,
        "latency_s": latency_s,
        "time_to_first_rows_s": time_to_first_rows_s,
    }

    mab_state = load_mab_state(_SERVER_MAB_STATE_PATH)
    record_outcome(ctx, format_used=format_used, outcome=outcome, mab_state=mab_state)
    save_mab_state(mab_state, _SERVER_MAB_STATE_PATH)

    return CallToolResult(
        content=[TextContent(type="text", text="Outcome recorded to server MAB state.")],
        structured_content={"recorded": True},
    )


@mcp.tool()
def validate_file(file_path: str, expected_type: Optional[str] = None) -> CallToolResult:
    """
    Validate a local file before use (blog-style file validation).

    This runs size, MIME, and hash checks. On success, it registers a
    validation id and returns a validated:// URI that can be used with
    the validated-file resource.

    Args:
        file_path: Path to the local file on disk.
        expected_type: Optional expected MIME type (e.g. image/png).
    """

    result = _validate_file_locally(file_path, expected_type)
    if not result.get("valid"):
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"File validation failed: {result.get('error')}",
                )
            ],
            structured_content=result,
        )

    validation_id = str(uuid.uuid4())
    VALIDATED_FILES[validation_id] = result

    validated_uri = f"validated://{validation_id}"
    descriptor: Dict[str, Any] = {
        "valid": True,
        "validation_id": validation_id,
        "validated_uri": validated_uri,
        "details": result.get("details", ""),
        "mime": result.get("mime"),
        "size": result.get("size"),
        "hash_prefix": str(result.get("hash", ""))[:16],
    }

    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=f"File validation passed: {descriptor['details']}",
            )
        ],
        structured_content=descriptor,
    )


@mcp.resource("validated://{validation_id}")
def read_validated_file(validation_id: str) -> bytes:
    """
    Resource reader that only exposes files that passed validate_file.

    The client should first call validate_file, then use the returned
    validated://{validation_id} URI with this resource.
    """

    meta = VALIDATED_FILES.get(validation_id)
    if not meta:
        raise ValueError("Access denied: file not validated or unknown validation id")

    file_path = meta.get("file_path")
    if not file_path:
        raise ValueError("Access denied: missing file path metadata")

    path = Path(str(file_path))
    if not path.is_file():
        raise ValueError("Access denied: validated file no longer exists")

    return path.read_bytes()


# ----- JWT-style access control tools ----------------------------------------------

JWT_SECRET_ENV = "MCP_JWT_SECRET"
JWT_ALG = "HS256"
SESSION_TTL_SECONDS = 3600


def _require_jwt_lib() -> None:
    if jwt is None:  # pragma: no cover - optional dependency
        raise RuntimeError("PyJWT is not installed; install PyJWT to use auth tools.")


def _get_jwt_secret() -> str:
    secret = os.environ.get(JWT_SECRET_ENV)
    if not secret:
        raise RuntimeError(
            f"{JWT_SECRET_ENV} is not set; configure a secret key to use auth tools."
        )
    return secret


def _create_session_token(username: str) -> str:
    _require_jwt_lib()
    secret = _get_jwt_secret()
    now = int(time.time())
    payload = {
        "username": username,
        "iat": now,
        "exp": now + SESSION_TTL_SECONDS,
    }
    token = jwt.encode(payload, secret, algorithm=JWT_ALG)  # type: ignore[arg-type]
    ACTIVE_SESSIONS[token] = username
    return token


def _verify_session_token(token: str) -> bool:
    _require_jwt_lib()
    secret = _get_jwt_secret()
    try:
        payload = jwt.decode(token, secret, algorithms=[JWT_ALG])  # type: ignore[arg-type]
    except Exception:
        return False

    if token not in ACTIVE_SESSIONS:
        return False

    exp = payload.get("exp")
    if isinstance(exp, (int, float)) and exp <= time.time():
        return False

    return True


def _get_username_from_token(token: str) -> Optional[str]:
    return ACTIVE_SESSIONS.get(token)


def _validate_credentials(username: str, credentials: str) -> bool:
    """
    Extremely simple credential check for demo purposes.

    In real deployments, integrate with your identity provider.
    """

    env_user = os.environ.get("MCP_DEMO_USERNAME", "demo")
    env_pass = os.environ.get("MCP_DEMO_PASSWORD", "password")
    return username == env_user and credentials == env_pass


def _check_resource_permission(username: str, resource: str, operation: str) -> bool:
    """
    Simple RBAC: demo user can do anything, others read-only on /public/.
    """

    if username == os.environ.get("MCP_DEMO_USERNAME", "demo"):
        return True

    if resource.startswith("/public/") and operation == "read":
        return True

    return False


@mcp.tool()
def authenticate_client(username: str, credentials: str) -> CallToolResult:
    """
    Blog-style authenticate_client: issue a short-lived session token (JWT).

    The token can then be passed to access_protected_resource for
    fine-grained authorization at the MCP tool layer.
    """

    if not _validate_credentials(username, credentials):
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text="Authentication failed",
                )
            ],
            structured_content={"authenticated": False},
        )

    token = _create_session_token(username)
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=f"Authentication successful. Session token: {token}",
            )
        ],
        structured_content={"authenticated": True, "session_token": token},
    )


@mcp.tool()
def access_protected_resource(
    session_token: str,
    resource_path: str,
    operation: str = "read",
) -> CallToolResult:
    """
    Blog-style access_protected_resource using JWT + simple RBAC.
    """

    if not _verify_session_token(session_token):
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text="Access denied: invalid session token",
                )
            ],
            structured_content={"authorized": False},
        )

    username = _get_username_from_token(session_token) or "<unknown>"
    if not _check_resource_permission(username, resource_path, operation):
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=(
                        f"Access denied: insufficient permissions for {operation} "
                        f"on {resource_path}"
                    ),
                )
            ],
            structured_content={"authorized": False, "username": username},
        )

    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=f"Access granted for {operation} on {resource_path}",
            )
        ],
        structured_content={"authorized": True, "username": username},
    )


# ----- Data privacy tools (PII + encryption) ---------------------------------------

PII_PATTERNS: Dict[str, re.Pattern[str]] = {
    "ssn": re.compile(r"\\b\\d{3}-\\d{2}-\\d{4}\\b"),
    "email": re.compile(
        r"\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b",
        re.IGNORECASE,
    ),
    "phone": re.compile(r"\\b\\d{3}-\\d{3}-\\d{4}\\b"),
    "credit_card": re.compile(
        r"\\b\\d{4}[-\\s]?\\d{4}[-\\s]?\\d{4}[-\\s]?\\d{4}\\b",
    ),
}

FERNET_KEY_ENV = "MCP_ENCRYPTION_KEY"


def _get_fernet() -> Fernet:
    if Fernet is None:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "cryptography is not installed; install cryptography to use encryption tools."
        )

    key = os.environ.get(FERNET_KEY_ENV)
    if not key:
        # For demo purposes we lazily generate a key if not set.
        generated = Fernet.generate_key()
        os.environ[FERNET_KEY_ENV] = generated.decode("utf-8")
        key_bytes = generated
    else:
        key_bytes = key.encode("utf-8")

    return Fernet(key_bytes)


@mcp.tool()
def scan_for_pii(text_content: str) -> CallToolResult:
    """
    Blog-style PII scanner: detect common PII patterns and return a sanitized summary.
    """

    findings: Dict[str, int] = {}
    for pii_type, pattern in PII_PATTERNS.items():
        matches = pattern.findall(text_content)
        if matches:
            findings[pii_type] = len(matches)

    if not findings:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text="No PII detected in content",
                )
            ],
            structured_content={"pii_found": False, "findings": {}},
        )

    sanitized_text = text_content
    for pii_type, pattern in PII_PATTERNS.items():
        if pii_type in findings:
            sanitized_text = pattern.sub(
                f"[REDACTED_{pii_type.upper()}]", sanitized_text
            )

    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=f"PII detected and sanitized. Found: {list(findings.keys())}",
            )
        ],
        structured_content={
            "pii_found": True,
            "findings": findings,
            "sanitized_text": sanitized_text,
        },
    )


@mcp.tool()
def encrypt_sensitive_data(data_content: str) -> CallToolResult:
    """
    Blog-style encrypt_sensitive_data using Fernet symmetric encryption.
    """

    fernet = _get_fernet()
    encrypted = fernet.encrypt(data_content.encode("utf-8"))
    encrypted_b64 = encrypted.decode("utf-8", errors="ignore")

    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=f"Data encrypted successfully. Length: {len(encrypted)} bytes",
            )
        ],
        structured_content={
            "encrypted": encrypted_b64,
            "length": len(encrypted),
        },
    )


# ----- HTTP data plane endpoints --------------------------------------------------


def _get_config_or_404(result_id: str) -> ResultConfig:
    config = RESULT_REGISTRY.get(result_id)
    if config is None:
        return Response(
            content=json.dumps({"error": "unknown result_id"}),
            media_type="application/json",
            status_code=404,
        )  # type: ignore[return-value]
    return config


def _parquet_blob_data(config: ResultConfig, result_id: str) -> bytes:
    comp = config.compression or _get_default_compression()
    enc_strat = config.encoding_strategy or _get_default_encoding_strategy()
    if config.cached_parquet_blob_bytes is not None and config.cached_parquet_codec == (comp, enc_strat):
        # F7: serve pre-encoded Parquet bytes from registration.
        _payload_cache_touch(result_id)
        return config.cached_parquet_blob_bytes
    if config.materialized_path is not None:
        return _encode_parquet(_resolve_arrow_table(config), comp, enc_strat)
    return _get_parquet_blob_bytes(config.n_rows, config.n_cols, comp, enc_strat)


async def parquet_blob_endpoint(request) -> Response:
    """Return a single Parquet file representing the full dataset."""

    result_id: str = request.path_params["result_id"]
    config = _get_config_or_404(result_id)
    if isinstance(config, Response):  # error case
        return config

    comp = config.compression or _get_default_compression()
    enc_strat = config.encoding_strategy or _get_default_encoding_strategy()
    data = _parquet_blob_data(config, result_id)

    headers = {
        "X-Benchmark-Rows": str(config.n_rows),
        "X-Benchmark-Cols": str(config.n_cols),
        "X-Benchmark-Bytes": str(len(data)),
        "X-Benchmark-Compression": comp,
        "X-Benchmark-Encoding-Strategy": enc_strat,
    }

    return Response(
        content=data,
        media_type="application/octet-stream",
        headers=headers,
    )


def _stream_parquet_chunks(config: ResultConfig):
    """
    Yield a sequence of length-prefixed micro-Parquet chunks.

    Each chunk is:
        [8-byte big-endian length][parquet-bytes]
    """

    assert config.rows_per_chunk is not None
    rows_per_chunk = config.rows_per_chunk
    comp = config.compression or _get_default_compression()
    enc_strat = config.encoding_strategy or _get_default_encoding_strategy()
    offset = 0

    if config.materialized_path is not None or config.cached_dataframe is not None:
        # F7: prefer the cached DataFrame; fall through to disk read when missing.
        # Slice the cached Arrow table (zero-copy) and derive the encodings once; converting a DataFrame slice for
        # every chunk cost about 90 ms per 8K-row chunk of a 34-column table.
        full_table = _resolve_arrow_table(config)
        params = arms.encoding_params(full_table, enc_strat)
        total_rows = full_table.num_rows
        while offset < total_rows:
            this_rows = min(rows_per_chunk, total_rows - offset)
            chunk = _encode_parquet(full_table.slice(offset, this_rows), comp, enc_strat, params)
            length_prefix = len(chunk).to_bytes(8, byteorder="big")
            yield length_prefix + chunk
            offset += this_rows
    else:
        total_rows = config.n_rows
        while offset < total_rows:
            this_rows = min(rows_per_chunk, total_rows - offset)
            chunk = _get_parquet_chunk_bytes(
                config.n_rows, config.n_cols, offset, this_rows, comp, enc_strat,
            )
            length_prefix = len(chunk).to_bytes(8, byteorder="big")
            yield length_prefix + chunk
            offset += this_rows


async def parquet_stream_endpoint(request) -> StreamingResponse:
    """Return a streaming response of length-prefixed micro-Parquet chunks."""

    result_id: str = request.path_params["result_id"]
    config = _get_config_or_404(result_id)
    if isinstance(config, Response):  # error case
        # Wrap the error response in a StreamingResponse for consistency
        return StreamingResponse(
            iter([config.body]),
            status_code=config.status_code,
            media_type="application/json",
        )

    headers = {
        "X-Benchmark-Rows": str(config.n_rows),
        "X-Benchmark-Cols": str(config.n_cols),
        "X-Benchmark-Rows-Per-Chunk": str(config.rows_per_chunk or 0),
    }

    return StreamingResponse(
        _stream_parquet_chunks(config),
        media_type="application/octet-stream",
        headers=headers,
    )


def _ipc_blob_data(config: ResultConfig, result_id: str) -> bytes:
    ipc_comp = config.ipc_compression or _get_default_arrow_ipc_compression()
    if ipc_comp not in VALID_ARROW_IPC_COMPRESSIONS:
        ipc_comp = "none"
    if config.cached_arrow_ipc_blob_bytes is not None and config.cached_arrow_ipc_codec == ipc_comp:
        # F7: serve pre-encoded Arrow IPC bytes from registration.
        _payload_cache_touch(result_id)
        return config.cached_arrow_ipc_blob_bytes
    if config.materialized_path is not None:
        return _encode_arrow_ipc_file(_resolve_arrow_table(config), ipc_comp)
    return _get_arrow_ipc_blob_bytes(config.n_rows, config.n_cols, ipc_comp)


async def ipc_blob_endpoint(request) -> Response:
    """Return a single Arrow IPC file representing the full dataset."""

    result_id: str = request.path_params["result_id"]
    config = _get_config_or_404(result_id)
    if isinstance(config, Response):
        return config

    ipc_comp = config.ipc_compression or _get_default_arrow_ipc_compression()
    if ipc_comp not in VALID_ARROW_IPC_COMPRESSIONS:
        ipc_comp = "none"
    data = _ipc_blob_data(config, result_id)

    headers = {
        "X-Benchmark-Rows": str(config.n_rows),
        "X-Benchmark-Cols": str(config.n_cols),
        "X-Benchmark-Bytes": str(len(data)),
        "X-Benchmark-IPC-Compression": ipc_comp,
    }

    return Response(
        content=data,
        media_type="application/vnd.apache.arrow.file",
        headers=headers,
    )


def _stream_arrow_ipc_chunks(config: ResultConfig):
    """
    Yield length-prefixed micro-IPC file chunks (same framing as Parquet stream).

    Each chunk is [8-byte big-endian length][IPC file bytes for that slice].
    """

    assert config.rows_per_chunk is not None
    rows_per_chunk = config.rows_per_chunk
    ipc_comp = config.ipc_compression or _get_default_arrow_ipc_compression()
    if ipc_comp not in VALID_ARROW_IPC_COMPRESSIONS:
        ipc_comp = "none"
    offset = 0

    if config.materialized_path is not None or config.cached_dataframe is not None:
        full_table = _resolve_arrow_table(config)
        total_rows = full_table.num_rows
        while offset < total_rows:
            this_rows = min(rows_per_chunk, total_rows - offset)
            chunk = _encode_arrow_ipc_file(full_table.slice(offset, this_rows), ipc_comp)
            length_prefix = len(chunk).to_bytes(8, byteorder="big")
            yield length_prefix + chunk
            offset += this_rows
    else:
        total_rows = config.n_rows
        while offset < total_rows:
            this_rows = min(rows_per_chunk, total_rows - offset)
            chunk = _get_arrow_ipc_chunk_bytes(
                config.n_rows, config.n_cols, offset, this_rows, ipc_comp,
            )
            length_prefix = len(chunk).to_bytes(8, byteorder="big")
            yield length_prefix + chunk
            offset += this_rows


async def ipc_stream_endpoint(request) -> StreamingResponse:
    """Return length-prefixed Arrow IPC file chunks (same framing as Parquet stream)."""

    result_id: str = request.path_params["result_id"]
    config = _get_config_or_404(result_id)
    if isinstance(config, Response):
        return StreamingResponse(
            iter([config.body]),
            status_code=config.status_code,
            media_type="application/json",
        )

    ipc_comp = config.ipc_compression or _get_default_arrow_ipc_compression()
    headers = {
        "X-Benchmark-Rows": str(config.n_rows),
        "X-Benchmark-Cols": str(config.n_cols),
        "X-Benchmark-Rows-Per-Chunk": str(config.rows_per_chunk or 0),
        "X-Benchmark-IPC-Compression": ipc_comp,
    }

    return StreamingResponse(
        _stream_arrow_ipc_chunks(config),
        media_type="application/octet-stream",
        headers=headers,
    )


async def raw_blob_endpoint(request) -> Response:
    """Return raw bytes for an unstructured registered result."""
    result_id: str = request.path_params["result_id"]
    config = _get_config_or_404(result_id)
    if isinstance(config, Response):
        return config
    if config.payload_kind != "unstructured" or config.raw_path is None:
        return Response(
            content=json.dumps({"error": "result_id is not unstructured"}),
            media_type="application/json",
            status_code=400,
        )
    data = config.raw_path.read_bytes()
    headers = {"X-Benchmark-Bytes": str(len(data))}
    return Response(
        content=data,
        media_type=config.raw_mime_type or "application/octet-stream",
        headers=headers,
    )


# ----- Unstructured payload codecs ------------------------------------------------
# Byte-codec blob arms next to gzip_blob. They carry no Content-Encoding header so a client without brotli/zstd
# libraries is not surprised by automatic decoding; the codec and uncompressed size travel in headers/descriptor.
# arm -> (pyarrow codec, level, route, hint key)
RAW_BYTE_CODECS: Dict[str, Tuple[str, int, str, str]] = {
    "zstd_blob": ("zstd", 3, "raw-zstd", "zstd_bytes"),
    "brotli_blob": ("brotli", 5, "raw-brotli", "brotli_bytes"),
}
_SAMPLE_CHECK_ABOVE = 1024 * 1024  # payloads larger than this are sampled before compressing
_SAMPLE_BYTES = 256 * 1024
_INCOMPRESSIBLE_RATIO = 0.97  # a sample that compresses less than this is treated as already compressed


def _compress_raw(data: bytes, codec: str, level: Optional[int]) -> bytes:
    return pa.Codec(codec, compression_level=level).compress(data, asbytes=True)


def _raw_gzip_blob(cfg: "ResultConfig", result_id: str) -> bytes:
    if cfg.raw_gzip_path is None or not cfg.raw_gzip_path.is_file():
        MATERIALIZED_RAW_DIR.mkdir(parents=True, exist_ok=True)
        gz_path = MATERIALIZED_RAW_DIR / f"{result_id}.bin.gz"
        gz_path.write_bytes(gzip.compress(cfg.raw_path.read_bytes()))
        cfg.raw_gzip_path = gz_path
    return cfg.raw_gzip_path.read_bytes()


def _raw_codec_blob(cfg: "ResultConfig", result_id: str, arm: str) -> bytes:
    """Compressed payload for a byte-codec arm, cached on disk beside the raw file."""
    codec, level, _, _ = RAW_BYTE_CODECS[arm]
    path = cfg.raw_codec_paths.get(arm)
    if path is None or not path.is_file():
        MATERIALIZED_RAW_DIR.mkdir(parents=True, exist_ok=True)
        path = MATERIALIZED_RAW_DIR / f"{result_id}.{codec}"
        path.write_bytes(_compress_raw(cfg.raw_path.read_bytes(), codec, level))
        cfg.raw_codec_paths[arm] = path
    return path.read_bytes()


def _unstructured_hints(cfg: "ResultConfig", result_id: str, cost_aware_mbps: float = 0.0) -> Dict[str, Any]:
    """Size of every unstructured arm, for the selector.

    With cost_aware_mbps > 0 (FORMAT_COST_AWARE=1, min_latency, known link speed) the sizes are estimates from a
    sample, nothing is compressed or written, and the hints carry per-arm compute times; see cost_model.py.

    Compressed sizes are exact. For payloads over 1 MB the first 256 KB is compressed first; if it barely
    shrinks (images, audio and other already-compressed media) the codec arms are left out so raw wins,
    instead of spending seconds compressing something that will not get smaller.
    """
    raw_bytes = cfg.raw_path.stat().st_size
    hints: Dict[str, Any] = {
        "raw_bytes": int(raw_bytes),
        "gzip_bytes": None,
        "zstd_bytes": None,
        "brotli_bytes": None,
        "text_inline_bytes": int(raw_bytes) if raw_bytes <= MAX_INLINE_TEXT_BYTES else None,
        "already_compressed": False,
    }
    if cost_aware_mbps > 0:
        if raw_bytes <= _json_obvious_winner_bytes():  # the selector returns tiny text inline without looking at sizes
            return hints
        def read_sample() -> bytes:
            with open(cfg.raw_path, "rb") as f:
                return f.read(cost_model.SAMPLE_BYTES)
        hints.update(cost_model.estimate_hints(read_sample, raw_bytes, cost_aware_mbps))
        return hints
    if raw_bytes > _SAMPLE_CHECK_ABOVE:
        sample = cfg.raw_path.read_bytes()[:_SAMPLE_BYTES]
        if len(_compress_raw(sample, "zstd", 1)) > _INCOMPRESSIBLE_RATIO * len(sample):
            hints["already_compressed"] = True
            return hints
    hints["gzip_bytes"] = len(_raw_gzip_blob(cfg, result_id))
    for arm, (_, _, _, key) in RAW_BYTE_CODECS.items():
        hints[key] = len(_raw_codec_blob(cfg, result_id, arm))
    return hints


async def raw_gzip_blob_endpoint(request) -> Response:
    """Return gzip-compressed raw bytes for an unstructured result."""
    result_id: str = request.path_params["result_id"]
    config = _get_config_or_404(result_id)
    if isinstance(config, Response):
        return config
    if config.payload_kind != "unstructured" or config.raw_path is None:
        return Response(
            content=json.dumps({"error": "result_id is not unstructured"}),
            media_type="application/json",
            status_code=400,
        )
    gz = _raw_gzip_blob(config, result_id)
    headers = {"X-Benchmark-Bytes": str(len(gz)), "Content-Encoding": "gzip"}
    return Response(
        content=gz,
        media_type=config.raw_mime_type or "application/octet-stream",
        headers=headers,
    )


def _raw_codec_endpoint(arm: str):
    """Endpoint factory for the zstd/brotli blob arms."""
    codec = RAW_BYTE_CODECS[arm][0]

    async def endpoint(request) -> Response:
        result_id: str = request.path_params["result_id"]
        config = _get_config_or_404(result_id)
        if isinstance(config, Response):
            return config
        if config.payload_kind != "unstructured" or config.raw_path is None:
            return Response(
                content=json.dumps({"error": "result_id is not unstructured"}),
                media_type="application/json",
                status_code=400,
            )
        blob = _raw_codec_blob(config, result_id, arm)
        return Response(
            content=blob,
            media_type="application/octet-stream",
            headers={
                "X-Benchmark-Bytes": str(len(blob)),
                "X-Payload-Codec": codec,
                "X-Uncompressed-Bytes": str(config.raw_path.stat().st_size),
                "X-Original-Content-Type": config.raw_mime_type or "application/octet-stream",
            },
        )

    return endpoint


# ----- Arm registry surface: descriptor fields, GET /data/{handle}, fetch_arm, list_arms --------


def _arm_codec(spec: "arms.ArmSpec", cfg: ResultConfig) -> Optional[str]:
    """Codec a handle's bytes are encoded with; arms that follow the process default read it from the config."""
    if spec.codec is not None:
        return spec.codec
    if spec.container == "parquet":
        return cfg.compression or _get_default_compression()
    if spec.container == "arrow_ipc":
        return cfg.ipc_compression or _get_default_arrow_ipc_compression()
    return None


def _cached_blob(spec: "arms.ArmSpec", cfg: ResultConfig) -> Optional[bytes]:
    """The encoded bytes for a blob handle if registration already produced them (so no encoding is needed)."""
    if spec.name == "parquet_blob":
        key = (cfg.compression or _get_default_compression(), cfg.encoding_strategy or _get_default_encoding_strategy())
        if cfg.cached_parquet_codec == key:
            return cfg.cached_parquet_blob_bytes
    if spec.name == "arrow_ipc_blob" and cfg.cached_arrow_ipc_codec == _arm_codec(spec, cfg):
        return cfg.cached_arrow_ipc_blob_bytes
    return None


def _add_arm_fields(descriptor: Dict[str, Any], arm: str) -> Dict[str, Any]:
    """Add the registry fields to a descriptor in place; existing fields are left alone.

    size and digest are included only when the encoded bytes are already in memory, so a tool call never encodes.
    """
    spec = REGISTRY.get(arm)
    cfg = RESULT_REGISTRY[descriptor["id"]]
    descriptor.setdefault("format", arm)
    descriptor.setdefault("modality", spec.modality)
    descriptor.setdefault("mime_type", spec.mime_type)
    descriptor["codec"] = _arm_codec(spec, cfg)
    descriptor["data_url"] = f"http://localhost:8000/data/{descriptor['id']}"
    cached = _cached_blob(spec, cfg)
    if cached is not None:
        descriptor["size"] = len(cached)
        descriptor["digest"] = hashlib.sha256(cached).hexdigest()
    return descriptor


def _serve_table_codec_arm(cfg: ResultConfig, spec: "arms.ArmSpec") -> Response:
    table = _resolve_arrow_table(cfg)
    if spec.container == "parquet":
        data = arms.encode_parquet(table, spec.codec, spec.level, cfg.encoding_strategy or _get_default_encoding_strategy())
    else:
        data = arms.encode_arrow_ipc(table, spec.codec)
    return Response(
        content=data,
        media_type=spec.mime_type,
        headers={"X-Benchmark-Rows": str(cfg.n_rows), "X-Benchmark-Cols": str(cfg.n_cols), "X-Benchmark-Bytes": str(len(data))},
    )


async def data_endpoint(request) -> Response:
    """One route for every minted handle: dispatch on the arm it was minted for."""
    handle: str = request.path_params["handle"]
    cfg = _get_config_or_404(handle)
    if isinstance(cfg, Response):
        return cfg
    if cfg.arm is None or REGISTRY.get(cfg.arm).transport == "inline":
        return Response(
            content=json.dumps({"error": "handle was not minted for a fetchable arm; use a tool to get a descriptor"}),
            media_type="application/json",
            status_code=400,
        )
    spec = REGISTRY.get(cfg.arm)
    request.scope["path_params"] = {"result_id": handle}
    legacy = _LEGACY_ENDPOINTS.get(spec.name)
    response = await legacy(request) if legacy else _serve_table_codec_arm(cfg, spec)
    if isinstance(response, StreamingResponse):
        return response
    response.headers["X-Payload-Codec"] = response.headers.get("X-Payload-Codec") or str(_arm_codec(spec, cfg))
    response.headers["X-Content-Digest"] = hashlib.sha256(response.body).hexdigest()
    return response


_LEGACY_ENDPOINTS = {
    "parquet_blob": parquet_blob_endpoint,
    "parquet_stream": parquet_stream_endpoint,
    "arrow_ipc_blob": ipc_blob_endpoint,
    "arrow_ipc_stream": ipc_stream_endpoint,
    "raw_blob": raw_blob_endpoint,
    "gzip_blob": raw_gzip_blob_endpoint,
    "zstd_blob": _raw_codec_endpoint("zstd_blob"),
    "brotli_blob": _raw_codec_endpoint("brotli_blob"),
}


@mcp.tool()
def list_arms() -> CallToolResult:
    """List every registered arm: modality, container, codec, level and whether the selector may choose it."""
    arm_list = [spec.describe() for spec in REGISTRY.all()]
    return CallToolResult(
        content=[TextContent(type="text", text=f"{len(arm_list)} arms registered")],
        structured_content={"arms": arm_list},
    )


@mcp.tool()
def fetch_arm(arm: str, result_id: str) -> CallToolResult:
    """Return a descriptor for *result_id* delivered through a specific blob arm, selectable or not."""
    try:
        spec = REGISTRY.get(arm)
    except KeyError:
        raise ValueError(f"Unknown arm: {arm}") from None
    if spec.transport != "blob":
        raise ValueError(f"{arm} is a {spec.transport} arm; fetch_arm serves blob arms (use the large_* tools for streams)")
    src = RESULT_REGISTRY.get(result_id)
    if src is None:
        raise ValueError(f"Unknown result_id: {result_id}")
    if (spec.modality == "table") != (src.payload_kind == "tabular"):
        raise ValueError(f"{arm} is a {spec.modality} arm but {result_id} is {src.payload_kind}")
    if arm == "parquet_blob":
        return large_parquet_blob(0, 0, result_id=result_id)
    if arm == "arrow_ipc_blob":
        return large_arrow_ipc_blob(0, 0, result_id=result_id)

    new_id = str(uuid.uuid4())
    if spec.modality == "table":
        RESULT_REGISTRY[new_id] = ResultConfig(
            n_rows=src.n_rows,
            n_cols=src.n_cols,
            compression=spec.codec if spec.container == "parquet" else None,
            encoding_strategy=_get_default_encoding_strategy() if spec.container == "parquet" else None,
            ipc_compression=spec.codec if spec.container == "arrow_ipc" else None,
            materialized_path=src.materialized_path,
            arm=arm,
        )
        _inherit_caches(new_id, result_id)
        url = f"http://localhost:8000/data/{new_id}"
        descriptor: Dict[str, Any] = {"mode": arm, "id": new_id, "url": url, "n_rows": src.n_rows, "n_cols": src.n_cols}
    else:
        RESULT_REGISTRY[new_id] = ResultConfig(
            n_rows=0,
            n_cols=0,
            payload_kind="unstructured",
            raw_path=src.raw_path,
            raw_mime_type=src.raw_mime_type,
            raw_charset=src.raw_charset,
            raw_gzip_path=src.raw_gzip_path,
            raw_codec_paths=src.raw_codec_paths,
            arm=arm,
        )
        route = {"raw_blob": "raw", "gzip_blob": "raw-gzip", "zstd_blob": "raw-zstd", "brotli_blob": "raw-brotli"}[arm]
        descriptor = {"mode": arm, "id": new_id, "url": f"http://localhost:8000/{route}/{new_id}",
                      "mime_type": src.raw_mime_type or "application/octet-stream"}
    _add_arm_fields(descriptor, arm)
    return CallToolResult(
        content=[TextContent(type="text", text=f"{arm} descriptor: {descriptor['data_url']}")],
        structured_content=descriptor,
        _meta={"result_id": new_id},
    )


# ----- Materialized result registration (HTTP POST) --------------------------------


async def register_materialized_endpoint(request) -> Response:
    """Accept a Parquet file upload and register it for benchmarking."""
    content_type = request.headers.get("content-type", "")
    if "application/octet-stream" not in content_type and "multipart" not in content_type:
        return Response(
            content=json.dumps({"error": "Content-Type must be application/octet-stream"}),
            media_type="application/json",
            status_code=400,
        )

    body = await request.body()
    if not body:
        return Response(
            content=json.dumps({"error": "empty body"}),
            media_type="application/json",
            status_code=400,
        )

    try:
        pf = pq.ParquetFile(BytesIO(body))
        meta = pf.metadata
        n_rows = meta.num_rows
        n_cols = meta.num_columns
    except Exception as exc:
        return Response(
            content=json.dumps({"error": f"invalid Parquet file: {exc}"}),
            media_type="application/json",
            status_code=400,
        )

    if n_rows > MAX_MATERIALIZED_ROWS:
        return Response(
            content=json.dumps({
                "error": f"too many rows ({n_rows}); limit is {MAX_MATERIALIZED_ROWS}",
            }),
            media_type="application/json",
            status_code=400,
        )

    MATERIALIZED_DIR.mkdir(parents=True, exist_ok=True)
    result_id = str(uuid.uuid4())
    path = MATERIALIZED_DIR / f"{result_id}.parquet"
    path.write_bytes(body)

    cfg = ResultConfig(
        n_rows=n_rows,
        n_cols=n_cols,
        payload_kind="tabular",
        materialized_path=path,
    )

    # Fix 2 + F7: pre-compute size hints AND payload caches while the
    # Parquet body is still in memory. This eliminates the describe-time
    # re-read + encode round trip on the hot path AND lets every later
    # `large_json` / parquet/IPC blob fetch serve from cache.
    try:
        table = pf.read()  # uses the already-opened ParquetFile
        df = table.to_pandas()
        _populate_materialized_caches(
            cfg,
            result_id=result_id,
            df=df,
            table=table,
        )
    except Exception:
        # Pre-computation is best-effort; on failure the describe path falls
        # back to live computation (slower but correct).
        cfg.cached_hints = None

    RESULT_REGISTRY[result_id] = cfg

    return Response(
        content=json.dumps({
            "result_id": result_id,
            "n_rows": n_rows,
            "n_cols": n_cols,
        }),
        media_type="application/json",
        status_code=201,
    )


async def register_materialized_raw_endpoint(request) -> Response:
    """
    Accept a raw (unstructured) byte payload and register it for benchmarking.

    The client may set:
    - Content-Type: e.g. text/plain; charset=utf-8 or application/octet-stream
    """
    content_type = request.headers.get("content-type", "").strip()
    body = await request.body()
    if not body:
        return Response(
            content=json.dumps({"error": "empty body"}),
            media_type="application/json",
            status_code=400,
        )

    MATERIALIZED_RAW_DIR.mkdir(parents=True, exist_ok=True)
    result_id = str(uuid.uuid4())
    path = MATERIALIZED_RAW_DIR / f"{result_id}.bin"
    path.write_bytes(body)

    mime = content_type.split(";", 1)[0].strip() if content_type else "application/octet-stream"
    charset = None
    if "charset=" in content_type.lower():
        try:
            charset = content_type.split("charset=", 1)[1].strip()
        except Exception:
            charset = None

    RESULT_REGISTRY[result_id] = ResultConfig(
        n_rows=0,
        n_cols=0,
        payload_kind="unstructured",
        raw_path=path,
        raw_mime_type=mime or "application/octet-stream",
        raw_charset=charset,
    )

    return Response(
        content=json.dumps(
            {
                "result_id": result_id,
                "payload_kind": "unstructured",
                "mime_type": mime,
                "bytes": len(body),
            }
        ),
        media_type="application/json",
        status_code=201,
    )

# ----- Authentication (API key / Bearer token) -----------------------------------
# When MCP_API_KEY is set, all requests must include: Authorization: Bearer <MCP_API_KEY>
# When unset, no authentication is required (backward compatible).


class BearerAuthMiddleware:
    """
    Validate Bearer token against MCP_API_KEY env var.
    Protects MCP control plane and data plane (blobs, streams).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._expected_key: str | None = os.environ.get("MCP_API_KEY")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if self._expected_key is None or self._expected_key == "":
            await self.app(scope, receive, send)
            return

        auth_header = None
        for name, value in scope.get("headers", []):
            if name.lower() == b"authorization":
                auth_header = value.decode("latin-1").strip()
                break

        if not auth_header or not auth_header.lower().startswith("bearer "):
            await self._send_401(send, "Missing or invalid Authorization header")
            return

        token = auth_header[7:].strip()
        if token != self._expected_key:
            await self._send_401(send, "Invalid API key")
            return

        await self.app(scope, receive, send)

    async def _send_401(self, send: Send, detail: str) -> None:
        body = json.dumps({"error": "unauthorized", "detail": detail}).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"www-authenticate", b'Bearer realm="MCP", error="invalid_token"'),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


# ----- Starlette app wiring -------------------------------------------------------


@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    """Run the MCP session manager alongside Starlette."""

    async with mcp.session_manager.run():
        yield


# Opt-in pyinstrument middleware. Profiles each HTTP request that hits the ASGI app
# and writes a per-request HTML flamegraph + speedscope JSON under
# results/profiling/server/. Enabled only when PYINSTRUMENT_PROFILE=1.
class PyinstrumentMiddleware:
    """ASGI middleware: per-request statistical profile via pyinstrument."""

    def __init__(self, app, out_dir: Path):
        self.app = app
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        from pyinstrument import Profiler  # local import keeps cold path cheap
        from pyinstrument.renderers import SpeedscopeRenderer

        method = scope.get("method", "GET")
        raw_path = scope.get("path", "/") or "/"
        # Detect MCP tool calls so the per-request file name carries the tool name.
        # MCP tool calls go through POST /mcp/mcp; we tag the body's "method" field
        # later by reading the wrapped body. Cheaper: attach a counter.
        self._counter += 1
        seq = self._counter
        safe_path = raw_path.strip("/").replace("/", "_") or "root"

        profiler = Profiler(async_mode="enabled", interval=0.0005)
        profiler.start()
        captured_tool: list[str] = []

        async def receive_wrapper():
            msg = await receive()
            if msg.get("type") == "http.request":
                body = msg.get("body", b"") or b""
                if body[:1] == b"{":
                    try:
                        obj = json.loads(body.decode("utf-8", errors="replace"))
                        if isinstance(obj, dict):
                            params = obj.get("params") or {}
                            name = (
                                params.get("name")
                                if isinstance(params, dict)
                                else None
                            )
                            if isinstance(name, str) and name:
                                captured_tool.append(name)
                            elif isinstance(obj.get("method"), str):
                                captured_tool.append(obj["method"])
                    except Exception:
                        pass
            return msg

        try:
            await self.app(scope, receive_wrapper, send)
        finally:
            profiler.stop()
            tool = captured_tool[0] if captured_tool else ""
            tool_tag = (
                "_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", tool)[:60] if tool else ""
            )
            stem = f"{seq:05d}_{method}_{safe_path}{tool_tag}"
            html_path = self.out_dir / f"{stem}.html"
            try:
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(profiler.output_html())
            except Exception:
                pass
            try:
                speedscope_path = self.out_dir / f"{stem}.speedscope.json"
                with open(speedscope_path, "w", encoding="utf-8") as f:
                    f.write(profiler.output(renderer=SpeedscopeRenderer()))
            except Exception:
                pass


_PROFILER_ENABLED = os.environ.get("PYINSTRUMENT_PROFILE", "").strip() == "1"
_PROFILER_OUT_DIR = Path(
    os.environ.get(
        "PYINSTRUMENT_OUT_DIR", "results/profiling/server"
    )
).resolve()

_middleware: list[Middleware] = [Middleware(BearerAuthMiddleware)]
if _PROFILER_ENABLED:
    _middleware.insert(
        0, Middleware(PyinstrumentMiddleware, out_dir=_PROFILER_OUT_DIR)
    )


app = Starlette(
    routes=[
        Mount(
            "/mcp",
            # In Docker runs, client reaches server as "server:8000"; use non-localhost
            # host setting so MCP transport does not enforce localhost-only Host headers.
            app=mcp.streamable_http_app(
                stateless_http=True,
                json_response=True,
                host="0.0.0.0",
            ),
        ),
        Route("/blobs/{result_id}.parquet", parquet_blob_endpoint, methods=["GET"]),
        Route("/streams/{result_id}", parquet_stream_endpoint, methods=["GET"]),
        Route("/ipc-blobs/{result_id}.arrow", ipc_blob_endpoint, methods=["GET"]),
        Route("/ipc-streams/{result_id}", ipc_stream_endpoint, methods=["GET"]),
        Route("/data/{handle}", data_endpoint, methods=["GET"]),
        Route("/materialized", register_materialized_endpoint, methods=["POST"]),
        Route("/materialized-raw", register_materialized_raw_endpoint, methods=["POST"]),
        Route("/raw/{result_id}", raw_blob_endpoint, methods=["GET"]),
        Route("/raw-gzip/{result_id}", raw_gzip_blob_endpoint, methods=["GET"]),
        Route("/raw-zstd/{result_id}", _raw_codec_endpoint("zstd_blob"), methods=["GET"]),
        Route("/raw-brotli/{result_id}", _raw_codec_endpoint("brotli_blob"), methods=["GET"]),
    ],
    middleware=_middleware,
    lifespan=lifespan,
)

