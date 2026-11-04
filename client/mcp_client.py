"""
MCP client for the benchmark server.

Provides a thin wrapper over streamable HTTP MCP: connect, call tools,
and fetch blob/stream URLs. Used by both the benchmark runner and any
LLM-backed workflow.
"""

from __future__ import annotations

import base64
import hashlib

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Iterable, Optional
from urllib.parse import urlsplit

# Normalized hints dict for server-side format selection responses
FormatHints = Dict[str, Any]

import httpx

from mcp import ClientSession  # type: ignore
from mcp.client.streamable_http import streamable_http_client  # type: ignore


# Env var for API key; when set, client sends Authorization: Bearer <key> on all requests
MCP_API_KEY_ENV = "MCP_API_KEY"

# Default MCP URL for the Java server (streamable HTTP at /mcp/mcp)
DEFAULT_MCP_URL = "http://localhost:8000/mcp/mcp"


class ForeignOriginError(Exception):
    """A request to an origin the client was not configured to talk to (descriptor URLs are untrusted input)."""


_LOOPBACK = ("localhost", "127.0.0.1", "[::1]")


def _origin(url: str) -> tuple:
    parts = urlsplit(url)
    host = parts.hostname or ""
    host = f"[{host}]" if ":" in host else host
    return parts.scheme, host, parts.port or (443 if parts.scheme == "https" else 80)


def allowed_origins(base_url: str, extra: Iterable[str] = ()) -> set:
    """Origins the client may contact: the MCP server's, any in `extra` or TRANSFORM_ALLOWED_ORIGINS, and, for a loopback
    server, the other loopback spellings of it (the Python server writes `localhost` into the URLs it mints)."""
    allowed = {_origin(base_url)}
    scheme, host, port = _origin(base_url)
    if host in _LOOPBACK:
        allowed |= {(scheme, h, port) for h in _LOOPBACK}
    configured = [o for o in os.environ.get("TRANSFORM_ALLOWED_ORIGINS", "").split(",") if o.strip()]
    allowed |= {_origin(o.strip()) for o in [*extra, *configured]}
    return allowed


def _strict_default() -> bool:
    return os.environ.get("TRANSFORM_SECURE", "").strip().lower() in ("1", "true", "yes")


def _origin_guard(allowed: set):
    async def refuse_foreign(request: httpx.Request) -> None:
        if _origin(str(request.url)) not in allowed:
            raise ForeignOriginError(f"refusing to contact {request.url.scheme}://{request.url.netloc.decode()}: "
                                     f"not the MCP server's origin")
    return refuse_foreign


def _unwrap_structured_content(data: Any) -> Any:
    """MCP may wrap non-object tool returns as {\"result\": ...}."""
    if isinstance(data, dict) and "result" in data and len(data) == 1:
        return data["result"]
    return data


def _auth_headers(api_key: Optional[str]) -> Dict[str, str]:
    """Build Authorization header when api_key is set (from arg or MCP_API_KEY env)."""
    key = api_key or os.environ.get(MCP_API_KEY_ENV)
    if not key or not key.strip():
        return {}
    return {"Authorization": f"Bearer {key.strip()}"}


@asynccontextmanager
async def connect(
    base_url: str = DEFAULT_MCP_URL,
    api_key: Optional[str] = None,
    strict_origins: Optional[bool] = None,
    extra_origins: Iterable[str] = (),
):
    """
    Connect to the MCP server. Use as: async with connect() as (session, client): ...

    When MCP_API_KEY is set (or api_key is passed), the client sends
    Authorization: Bearer <key> on MCP and on blob/stream fetches.

    With strict_origins (default: on when TRANSFORM_SECURE=1) the returned client raises ForeignOriginError instead of
    sending a request, and so also the key, to any origin other than the MCP server's (see allowed_origins).
    """
    headers = _auth_headers(api_key)
    strict = _strict_default() if strict_origins is None else strict_origins
    hooks = {"request": [_origin_guard(allowed_origins(base_url, extra_origins))]} if strict else {}

    if headers:
        async with httpx.AsyncClient(headers=headers, event_hooks=hooks) as http_client:
            async with streamable_http_client(base_url, http_client=http_client) as (
                read_stream,
                write_stream,
                *_rest,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session, http_client
    else:
        async with streamable_http_client(base_url) as (read_stream, write_stream, *_rest):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                async with httpx.AsyncClient(event_hooks=hooks) as client:
                    yield session, client


async def call_large_json(
    session: ClientSession,
    n_rows: int,
    n_cols: int,
    *,
    result_id: Optional[str] = None,
) -> Any:
    """Call large_json tool and return structured_content (list of records)."""
    args: Dict[str, Any] = {"n_rows": n_rows, "n_cols": n_cols}
    if result_id is not None:
        args["result_id"] = result_id
    result = await session.call_tool("large_json", arguments=args)
    return _unwrap_structured_content(result.structuredContent)


async def call_large_parquet_blob(
    session: ClientSession,
    n_rows: int,
    n_cols: int,
    *,
    result_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Call large_parquet_blob tool and return descriptor with url."""
    args: Dict[str, Any] = {"n_rows": n_rows, "n_cols": n_cols}
    if result_id is not None:
        args["result_id"] = result_id
    result = await session.call_tool("large_parquet_blob", arguments=args)
    desc = result.structuredContent or {}
    if not isinstance(desc.get("url"), str):
        raise RuntimeError(f"Unexpected descriptor from large_parquet_blob: {desc}")
    return desc


async def call_large_parquet_stream(
    session: ClientSession,
    n_rows: int,
    n_cols: int,
    rows_per_chunk: int,
    *,
    result_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Call large_parquet_stream tool and return descriptor with url."""
    args: Dict[str, Any] = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "rows_per_chunk": rows_per_chunk,
    }
    if result_id is not None:
        args["result_id"] = result_id
    result = await session.call_tool("large_parquet_stream", arguments=args)
    desc = result.structuredContent or {}
    if not isinstance(desc.get("url"), str):
        raise RuntimeError(f"Unexpected descriptor from large_parquet_stream: {desc}")
    return desc


async def call_large_arrow_ipc_blob(
    session: ClientSession,
    n_rows: int,
    n_cols: int,
    *,
    result_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Call large_arrow_ipc_blob tool and return descriptor with url."""
    args: Dict[str, Any] = {"n_rows": n_rows, "n_cols": n_cols}
    if result_id is not None:
        args["result_id"] = result_id
    result = await session.call_tool("large_arrow_ipc_blob", arguments=args)
    desc = result.structuredContent or {}
    if not isinstance(desc.get("url"), str):
        raise RuntimeError(f"Unexpected descriptor from large_arrow_ipc_blob: {desc}")
    return desc


async def call_large_arrow_ipc_stream(
    session: ClientSession,
    n_rows: int,
    n_cols: int,
    rows_per_chunk: int,
    *,
    result_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Call large_arrow_ipc_stream tool and return descriptor with url."""
    args: Dict[str, Any] = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "rows_per_chunk": rows_per_chunk,
    }
    if result_id is not None:
        args["result_id"] = result_id
    result = await session.call_tool("large_arrow_ipc_stream", arguments=args)
    desc = result.structuredContent or {}
    if not isinstance(desc.get("url"), str):
        raise RuntimeError(f"Unexpected descriptor from large_arrow_ipc_stream: {desc}")
    return desc


async def call_describe_result_formats(
    session: ClientSession,
    n_rows: int,
    n_cols: int,
    rows_per_chunk: int = 8192,
    *,
    result_id: Optional[str] = None,
    optimization_target: Optional[str] = None,
    prefer_streaming: bool = False,
) -> Optional[FormatHints]:
    """
    Call describe_result_formats hint tool; return normalized hints dict for
    format_selector.select_format_with_hints, or None if tool missing / error.

    When the server supports it, includes recommended_format, recommendation_target,
    recommendation_prefer_streaming (mirrors format_selector logic on the server).
    """
    try:
        args: Dict[str, Any] = {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "rows_per_chunk": rows_per_chunk,
            "prefer_streaming": prefer_streaming,
        }
        if result_id is not None:
            args["result_id"] = result_id
        if optimization_target is not None:
            args["optimization_target"] = optimization_target
        result = await session.call_tool(
            "describe_result_formats", arguments=args,
        )
    except Exception:
        return None
    data = result.structuredContent if result else None
    if not isinstance(data, dict):
        return None
    formats = data.get("formats")
    if not isinstance(formats, dict):
        return None

    # Unstructured blob hints (raw / gzip / optional inline text)
    raw_f = formats.get("raw_blob")
    gzip_f = formats.get("gzip_blob")
    text_f = formats.get("text_inline")
    if isinstance(raw_f, dict) and raw_f.get("approx_bytes") is not None:
        hints_u: FormatHints = {
            "raw_bytes": int(raw_f["approx_bytes"]),
        }
        if isinstance(gzip_f, dict) and gzip_f.get("approx_bytes") is not None:
            hints_u["gzip_bytes"] = int(gzip_f["approx_bytes"])
        if isinstance(text_f, dict) and text_f.get("approx_bytes") is not None:
            hints_u["text_inline_bytes"] = int(text_f["approx_bytes"])
        rec = data.get("recommended_format")
        if isinstance(rec, str) and rec:
            hints_u["recommended_format"] = rec
        rt = data.get("recommendation_target")
        if isinstance(rt, str) and rt:
            hints_u["recommendation_target"] = rt
        return hints_u

    json_f = formats.get("json")
    blob_f = formats.get("parquet_blob")
    stream_f = formats.get("parquet_stream")
    ipc_blob_f = formats.get("arrow_ipc_blob")
    ipc_stream_f = formats.get("arrow_ipc_stream")
    json_bytes = json_f.get("approx_bytes") if isinstance(json_f, dict) else None
    parquet_bytes = blob_f.get("approx_bytes") if isinstance(blob_f, dict) else None
    first_chunk = (
        stream_f.get("approx_first_chunk_bytes") if isinstance(stream_f, dict) else None
    )
    if json_bytes is None or parquet_bytes is None:
        return None
    hints: FormatHints = {
        "json_bytes": int(json_bytes),
        "parquet_bytes": int(parquet_bytes),
        "parquet_stream_first_chunk_bytes": int(first_chunk) if first_chunk is not None else None,
    }
    if isinstance(ipc_blob_f, dict) and ipc_blob_f.get("approx_bytes") is not None:
        hints["arrow_ipc_bytes"] = int(ipc_blob_f["approx_bytes"])
    if isinstance(ipc_stream_f, dict):
        ipc_fc = ipc_stream_f.get("approx_first_chunk_bytes")
        if ipc_fc is not None:
            hints["arrow_ipc_stream_first_chunk_bytes"] = int(ipc_fc)
    pq_comp = data.get("parquet_compression")
    pq_enc = data.get("parquet_encoding_strategy")
    ipc_comp = data.get("arrow_ipc_compression")
    if pq_comp is not None:
        hints["parquet_compression"] = str(pq_comp)
    if pq_enc is not None:
        hints["parquet_encoding_strategy"] = str(pq_enc)
    if ipc_comp is not None:
        hints["arrow_ipc_compression"] = str(ipc_comp)
    rec = data.get("recommended_format")
    if isinstance(rec, str) and rec:
        hints["recommended_format"] = rec
    rt = data.get("recommendation_target")
    if isinstance(rt, str) and rt:
        hints["recommendation_target"] = rt
    if "recommendation_prefer_streaming" in data:
        hints["recommendation_prefer_streaming"] = bool(
            data["recommendation_prefer_streaming"]
        )
    return hints


async def call_large_result_auto(
    session: ClientSession,
    n_rows: int,
    n_cols: int,
    rows_per_chunk: int = 8192,
    *,
    result_id: Optional[str] = None,
    optimization_target: Optional[str] = None,
    prefer_streaming: bool = False,
    use_mab: bool = False,
) -> Dict[str, Any]:
    """
    Call large_result_auto (one-shot server-side selection).
    Returns structured_content with payload_kind, chosen_format, payload, decode, etc.
    """
    args: Dict[str, Any] = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "rows_per_chunk": rows_per_chunk,
        "prefer_streaming": prefer_streaming,
        "use_mab": use_mab,
    }
    if result_id is not None:
        args["result_id"] = result_id
    if optimization_target is not None:
        args["optimization_target"] = optimization_target
    result = await session.call_tool("large_result_auto", arguments=args)
    data = result.structuredContent or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected structured_content from large_result_auto: {data}")
    inline = (getattr(result, "meta", None) or {}).get("transform/inline")
    if inline:  # the server sent the bytes in _meta instead of behind a URL (FORMAT_INLINE_MAX_BYTES)
        payload = base64.b64decode(inline["data"])
        if hashlib.sha256(payload).hexdigest() != inline["sha256"]:
            raise RuntimeError("inline payload failed its sha256 check")
        data = {**data, "inline_bytes": payload}
    return data


async def call_describe_and_fetch_result(
    session: ClientSession,
    n_rows: int,
    n_cols: int,
    rows_per_chunk: int = 8192,
    *,
    result_id: Optional[str] = None,
    optimization_target: Optional[str] = None,
    prefer_streaming: bool = False,
    use_mab: bool = False,
) -> Dict[str, Any]:
    """
    Call describe_and_fetch_result (fused describe + selection + payload).
    Same structured_content shape as large_result_auto.
    """
    args: Dict[str, Any] = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "rows_per_chunk": rows_per_chunk,
        "prefer_streaming": prefer_streaming,
        "use_mab": use_mab,
    }
    if result_id is not None:
        args["result_id"] = result_id
    if optimization_target is not None:
        args["optimization_target"] = optimization_target
    result = await session.call_tool("describe_and_fetch_result", arguments=args)
    data = result.structuredContent or {}
    if not isinstance(data, dict):
        raise RuntimeError(
            f"Unexpected structured_content from describe_and_fetch_result: {data}"
        )
    return data


async def call_bird_query_json(
    session: ClientSession,
    *,
    db_id: str,
    sql: str,
    max_rows: int = 500_000,
) -> Any:
    """Execute SQL on server and return inline JSON records."""
    args: Dict[str, Any] = {"db_id": db_id, "sql": sql, "max_rows": max_rows}
    result = await session.call_tool("bird_query_json", arguments=args)
    return result.structuredContent


async def call_bird_query_materialize(
    session: ClientSession,
    *,
    db_id: str,
    sql: str,
    max_rows: int = 500_000,
) -> Dict[str, Any]:
    """Execute SQL on server, materialize to Parquet, return {result_id,n_rows,n_cols}."""
    args: Dict[str, Any] = {"db_id": db_id, "sql": sql, "max_rows": max_rows}
    result = await session.call_tool("bird_query_materialize", arguments=args)
    data = result.structuredContent or {}
    # Some MCP transports wrap tool results as {"result": {...}}.
    if isinstance(data, dict) and "result" in data and isinstance(data.get("result"), dict):
        data = data["result"]
    if not isinstance(data, dict) or not data.get("result_id"):
        raise RuntimeError(f"Unexpected structured_content from bird_query_materialize: {data}")
    return data  # type: ignore[return-value]


async def call_bird_query_auto(
    session: ClientSession,
    *,
    db_id: str,
    sql: str,
    optimization_target: Optional[str] = None,
    rows_per_chunk: int = 8192,
    prefer_streaming: bool = False,
    use_mab: bool = False,
    max_rows: int = 500_000,
) -> Dict[str, Any]:
    """Execute SQL on server and return one-shot selected payload (large_result_auto shape)."""
    args: Dict[str, Any] = {
        "db_id": db_id,
        "sql": sql,
        "optimization_target": optimization_target,
        "rows_per_chunk": rows_per_chunk,
        "prefer_streaming": prefer_streaming,
        "use_mab": use_mab,
        "max_rows": max_rows,
    }
    # Drop None for cleanliness.
    args = {k: v for k, v in args.items() if v is not None}
    result = await session.call_tool("bird_query_auto", arguments=args)
    data = result.structuredContent or {}
    if isinstance(data, dict) and "result" in data and isinstance(data.get("result"), dict):
        data = data["result"]
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected structured_content from bird_query_auto: {data}")
    return data


async def call_bird_query_run_inline(
    session: ClientSession,
    *,
    db_id: str,
    sql: str,
    optimization_target: Optional[str] = None,
    rows_per_chunk: int = 8192,
    prefer_streaming: bool = False,
    use_mab: bool = False,
    max_rows: int = 500_000,
) -> Dict[str, Any]:
    """
    Round-2 (F9): execute SQL + select format + return payload in a single
    MCP round trip; matches large_result_auto's structured_content shape.

    Skips the parquet disk write entirely when JSON wins (BIRD common case).
    """
    args: Dict[str, Any] = {
        "db_id": db_id,
        "sql": sql,
        "optimization_target": optimization_target,
        "rows_per_chunk": rows_per_chunk,
        "prefer_streaming": prefer_streaming,
        "use_mab": use_mab,
        "max_rows": max_rows,
    }
    args = {k: v for k, v in args.items() if v is not None}
    result = await session.call_tool("bird_query_run_inline", arguments=args)
    data = result.structuredContent or {}
    if isinstance(data, dict) and "result" in data and isinstance(data.get("result"), dict):
        data = data["result"]
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected structured_content from bird_query_run_inline: {data}")
    return data


async def call_record_format_outcome(
    session: ClientSession,
    n_rows: int,
    n_cols: int,
    *,
    optimization_target: str,
    format_used: str,
    bytes: Optional[int] = None,
    latency_s: Optional[float] = None,
    time_to_first_rows_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Call record_format_outcome to update server-side MAB state."""
    args: Dict[str, Any] = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "optimization_target": optimization_target,
        "format_used": format_used,
        "bytes": bytes,
        "latency_s": latency_s,
        "time_to_first_rows_s": time_to_first_rows_s,
    }
    result = await session.call_tool("record_format_outcome", arguments=args)
    data = result.structuredContent or {}
    return data if isinstance(data, dict) else {}


async def register_materialized(
    client: httpx.AsyncClient,
    parquet_bytes: bytes,
    *,
    base_url: str = "http://localhost:8000",
) -> Dict[str, Any]:
    """POST a Parquet blob to /materialized and return {result_id, n_rows, n_cols}."""
    resp = await client.post(
        f"{base_url}/materialized",
        content=parquet_bytes,
        headers={"Content-Type": "application/octet-stream"},
    )
    resp.raise_for_status()
    return resp.json()


async def register_materialized_raw(
    client: httpx.AsyncClient,
    raw_bytes: bytes,
    *,
    base_url: str = "http://localhost:8000",
    content_type: str = "application/octet-stream",
) -> Dict[str, Any]:
    """POST raw bytes to /materialized-raw and return {result_id, payload_kind, mime_type, bytes}."""
    resp = await client.post(
        f"{base_url}/materialized-raw",
        content=raw_bytes,
        headers={"Content-Type": content_type},
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict) or not data.get("result_id"):
        raise RuntimeError(f"Unexpected response from register_materialized_raw: {data}")
    return data


async def fetch_blob(client: httpx.AsyncClient, url: str) -> bytes:
    """Download full blob from URL (e.g. Parquet file)."""
    resp = await client.get(url)
    resp.raise_for_status()
    return resp.content


async def fetch_stream_chunks(
    client: httpx.AsyncClient,
    url: str,
    *,
    length_prefix_bytes: int = 8,
    big_endian: bool = True,
) -> AsyncIterator[bytes]:
    """
    Stream from URL, yielding raw chunk bytes (without length prefix).
    Used for Parquet micro-chunks and Arrow IPC file chunks.
    Assumes length-prefixed format: [8-byte length][chunk bytes].
    """
    async with client.stream("GET", url) as resp:
        resp.raise_for_status()
        buffer = b""
        async for data in resp.aiter_bytes():
            buffer += data
            while len(buffer) >= length_prefix_bytes:
                chunk_len = int.from_bytes(
                    buffer[:length_prefix_bytes],
                    byteorder="big" if big_endian else "little",
                )
                if len(buffer) < length_prefix_bytes + chunk_len:
                    break
                chunk = buffer[length_prefix_bytes : length_prefix_bytes + chunk_len]
                buffer = buffer[length_prefix_bytes + chunk_len :]
                yield chunk
