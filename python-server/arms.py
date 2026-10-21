"""Arm registry: every way the server can deliver a result, and the encoders behind them.

An arm is (modality, container, codec, level, transport). Selection stays closed: the selector and the bandit only
see arms with ``selectable=True``; the rest are reachable through the ``fetch_arm`` tool and ``GET /data/{handle}``.

This module does not import ``server_app``; the server calls into it.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from codec_selector import select_encoding_params

PARQUET_MIME = "application/vnd.apache.parquet"
ARROW_MIME = "application/vnd.apache.arrow.file"
JSON_MIME = "application/json"


@dataclass(frozen=True)
class ArmSpec:
    name: str
    modality: str  # "table" | "unstructured"
    container: str  # "json" | "parquet" | "arrow_ipc" | "raw" | "text"
    transport: str  # "inline" | "blob" | "stream"
    codec: Optional[str] = None  # None: the server's process-wide default for the container
    level: Optional[int] = None
    lossless: bool = True
    selectable: bool = True

    @property
    def mime_type(self) -> str:
        return {"parquet": PARQUET_MIME, "arrow_ipc": ARROW_MIME, "json": JSON_MIME}.get(self.container, "application/octet-stream")

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "modality": self.modality, "container": self.container, "transport": self.transport,
                "codec": self.codec, "level": self.level, "lossless": self.lossless, "selectable": self.selectable,
                "mime_type": self.mime_type}


class ArmRegistry:
    def __init__(self) -> None:
        self._arms: Dict[str, ArmSpec] = {}

    def register(self, spec: ArmSpec) -> ArmSpec:
        if spec.name in self._arms:
            raise ValueError(f"arm already registered: {spec.name}")
        self._arms[spec.name] = spec
        return spec

    def get(self, name: str) -> ArmSpec:
        return self._arms[name]

    def names(self, selectable_only: bool = False) -> List[str]:
        return [n for n, s in self._arms.items() if s.selectable or not selectable_only]

    def for_modality(self, modality: str) -> List[ArmSpec]:
        return [s for s in self._arms.values() if s.modality == modality]

    def all(self) -> List[ArmSpec]:
        return list(self._arms.values())


def encoding_params(table: pa.Table, strategy: str) -> Optional[Dict[str, Any]]:
    """Per-column encodings for ``strategy="data_driven"`` (None otherwise); compute once to encode many slices."""
    return select_encoding_params(table) if strategy == "data_driven" else None


def encode_parquet(table: pa.Table, codec: str, level: Optional[int], strategy: str,
                   params: Optional[Dict[str, Any]] = None) -> bytes:
    """Parquet bytes for *table*; ``strategy="data_driven"`` picks dictionary and column encodings from the data.

    Pass *params* from ``encoding_params`` of the whole table to encode slices of it without re-deriving them.
    """
    buf = BytesIO()
    kwargs: Dict[str, Any] = {"compression": codec}
    if level is not None:
        kwargs["compression_level"] = level
    if strategy == "data_driven":
        params = params if params is not None else select_encoding_params(table)
        kwargs["use_dictionary"] = params.get("use_dictionary", [])
        if params.get("column_encoding"):
            kwargs["column_encoding"] = params["column_encoding"]
    pq.write_table(table, buf, **kwargs)
    return buf.getvalue()


def encode_arrow_ipc(table: pa.Table, codec: str) -> bytes:
    options = pa.ipc.IpcWriteOptions() if codec == "none" else pa.ipc.IpcWriteOptions(compression=codec)
    buf = pa.BufferOutputStream()
    with pa.ipc.new_file(buf, table.schema, options=options) as writer:
        writer.write_table(table)
    return buf.getvalue().to_pybytes()


def encode_raw(data: bytes, codec: str, level: Optional[int]) -> bytes:
    return pa.Codec(codec, compression_level=level).compress(data, asbytes=True)


REGISTRY = ArmRegistry()

for _spec in (
    ArmSpec("json", "table", "json", "inline"),
    ArmSpec("parquet_blob", "table", "parquet", "blob"),
    ArmSpec("parquet_stream", "table", "parquet", "stream"),
    ArmSpec("arrow_ipc_blob", "table", "arrow_ipc", "blob"),
    ArmSpec("arrow_ipc_stream", "table", "arrow_ipc", "stream"),
    ArmSpec("text_inline", "unstructured", "text", "inline"),
    ArmSpec("raw_blob", "unstructured", "raw", "blob", codec="none"),
    ArmSpec("gzip_blob", "unstructured", "raw", "blob", codec="gzip"),
    ArmSpec("zstd_blob", "unstructured", "raw", "blob", codec="zstd", level=3),
    ArmSpec("brotli_blob", "unstructured", "raw", "blob", codec="brotli", level=5),
    # Registered for explicit fetches (fetch_arm, /data); the selector and the bandit never choose them.
    ArmSpec("parquet_none_blob", "table", "parquet", "blob", codec="none", selectable=False),
    ArmSpec("parquet_gzip_blob", "table", "parquet", "blob", codec="gzip", selectable=False),
    ArmSpec("parquet_lz4_blob", "table", "parquet", "blob", codec="lz4", selectable=False),
    ArmSpec("parquet_zstd1_blob", "table", "parquet", "blob", codec="zstd", level=1, selectable=False),
    ArmSpec("parquet_zstd3_blob", "table", "parquet", "blob", codec="zstd", level=3, selectable=False),
    ArmSpec("parquet_zstd9_blob", "table", "parquet", "blob", codec="zstd", level=9, selectable=False),
    ArmSpec("arrow_ipc_lz4_blob", "table", "arrow_ipc", "blob", codec="lz4", selectable=False),
    ArmSpec("arrow_ipc_zstd_blob", "table", "arrow_ipc", "blob", codec="zstd", selectable=False),
):
    REGISTRY.register(_spec)
