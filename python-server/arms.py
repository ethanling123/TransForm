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


# Model-aware arms. Both are opt-in (selectable=False) and lossy for the original file: they hand a model the input it
# will in any case be given. Claude counts ceil(w / 28) * ceil(h / 28) visual tokens and downsizes above a long edge and
# token limit (1568 and 1568 for the standard tier), so pixels beyond that never reach the model; the resampler here
# is not the provider's, so equivalence is approximate. Speech models take 16 kHz mono audio.
VISION_MAX_EDGE = 1568
VISION_MAX_TOKENS = 1568


def fit_image_to_model(data: bytes, max_edge: int = VISION_MAX_EDGE, max_tokens: int = VISION_MAX_TOKENS) -> bytes:
    """PNG bytes of the image scaled down to the model's limits; the original bytes if it already fits or if scaling
    would not make the file smaller."""
    from PIL import Image

    im = Image.open(BytesIO(data))
    w, h = im.size

    def tokens(a: float, b: float) -> int:
        return -(-int(a) // 28) * -(-int(b) // 28)

    scale = min(1.0, max_edge / max(w, h))
    while tokens(w * scale, h * scale) > max_tokens:
        scale *= 0.98
    if scale >= 1.0:
        return data
    im = im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB")
    out = BytesIO()
    im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS).save(out, "PNG")
    # Flat graphics can compress better at full size than after resampling; the provider downscales either way.
    return out.getvalue() if out.tell() < len(data) else data


def audio_to_16k_mono(data: bytes) -> bytes:
    """16-bit 16 kHz mono WAV from any audio ffmpeg can read."""
    import shutil
    import subprocess
    import tempfile

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is needed for the audio_16k_blob arm")
    # Files, not pipes: a WAV written to a pipe cannot be seeked, so its header sizes would be wrong.
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = f"{tmp}/in", f"{tmp}/out.wav"
        with open(src, "wb") as f:
            f.write(data)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-ac", "1", "-ar", "16000",
                        "-c:a", "pcm_s16le", dst], check=True, capture_output=True)
        with open(dst, "rb") as f:
            return f.read()


MEDIA_TRANSFORMS = {"image_fit_blob": (fit_image_to_model, "image/png"), "audio_16k_blob": (audio_to_16k_mono, "audio/wav")}


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
    ArmSpec("image_fit_blob", "unstructured", "raw", "blob", codec="png-fit-to-model", lossless=False, selectable=False),
    ArmSpec("audio_16k_blob", "unstructured", "raw", "blob", codec="wav-16k-mono", lossless=False, selectable=False),
):
    REGISTRY.register(_spec)
