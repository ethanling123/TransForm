"""
Client package: MCP client and lightweight LLM for the benchmark workflow.
"""

from .mcp_client import (
    DEFAULT_MCP_URL,
    connect,
    call_large_json,
    call_large_parquet_blob,
    call_large_parquet_stream,
    call_large_arrow_ipc_blob,
    call_large_arrow_ipc_stream,
    call_describe_result_formats,
    fetch_blob,
    fetch_stream_chunks,
    register_materialized,
)
from .llm_client import (
    chat,
    complete,
    get_llm_backend,
    chat_ollama,
    chat_deepseek,
    chat_vertex,
)

__all__ = [
    "DEFAULT_MCP_URL",
    "connect",
    "call_large_json",
    "call_large_parquet_blob",
    "call_large_parquet_stream",
    "call_large_arrow_ipc_blob",
    "call_large_arrow_ipc_stream",
    "call_describe_result_formats",
    "fetch_blob",
    "fetch_stream_chunks",
    "register_materialized",
    "chat",
    "complete",
    "get_llm_backend",
    "chat_ollama",
    "chat_deepseek",
    "chat_vertex",
]
