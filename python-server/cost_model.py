"""Compute-time model for the cost-aware unstructured selector (experimental, off unless FORMAT_COST_AWARE=1).

The legacy selector counts only bytes, and learns the byte counts by compressing the whole payload with every codec.
This module lets the server decide with less work:

1. Break-even: if sending the payload with zero bytes left would still save less transfer time than the cheapest
   codec's compress-plus-decompress time, no codec can win, so the payload goes raw and is never sampled.
2. Staged sampling: otherwise compress the first 256 KB with zstd (the cheapest codec to sample); if zstd alone does not
   beat sending raw, go raw.
3. Only then sample gzip and brotli, and only those that could still beat zstd: a codec that takes longer than zstd
   by more than zstd's whole transfer time cannot win even if it shrank the payload to nothing.
4. The selector scores each arm as compute time + bytes / bandwidth, using the sampled sizes extrapolated to the payload.

Compute speeds come from a one-off microbenchmark of the codecs at the levels the server serves. Decompression speed
on the server stands in for the client's, which is not measured.
"""

from __future__ import annotations

import gzip
import os
import random
import time
from functools import lru_cache
from typing import Callable, Dict, Tuple

import pyarrow as pa

SAMPLE_BYTES = 256 * 1024
CALIBRATION_BYTES = 1024 * 1024


def _arrow(codec: str, level: int):
    c = pa.Codec(codec, compression_level=level)
    return (lambda b: c.compress(b, asbytes=True)), (lambda b, n: c.decompress(b, decompressed_size=n, asbytes=True))


# arm -> (compress, decompress(data, original_size)); levels match RAW_BYTE_CODECS and gzip.compress in server_app.
_zstd = _arrow("zstd", 3)
_brotli = _arrow("brotli", 5)
CODECS: Dict[str, Tuple[Callable[[bytes], bytes], Callable[[bytes, int], bytes]]] = {
    "zstd_blob": _zstd,
    "gzip_blob": (gzip.compress, lambda b, n: gzip.decompress(b)),
    "brotli_blob": _brotli,
}
SAMPLE_ORDER = ("zstd_blob", "gzip_blob", "brotli_blob")


def enabled() -> bool:
    return os.environ.get("FORMAT_COST_AWARE", "").strip().lower() in ("1", "true", "yes")


def _calibration_buffer() -> bytes:
    rng = random.Random(0)
    rows = (f'{{"id":{i},"name":"item-{rng.randrange(5000)}","price":{rng.random() * 100:.2f},"tags":["a","b"]}}\n'
            for i in range(40_000))
    return "".join(rows).encode()[:CALIBRATION_BYTES]


@lru_cache(maxsize=1)
def calibrate() -> Dict[str, Tuple[float, float]]:
    """arm -> (compress, decompress) seconds per byte, best of three on a text-like buffer."""
    buf = _calibration_buffer()
    out = {}
    for arm, (compress, decompress) in CODECS.items():
        best_c = best_d = float("inf")
        for _ in range(3):
            t = time.perf_counter()
            packed = compress(buf)
            best_c = min(best_c, time.perf_counter() - t)
            t = time.perf_counter()
            decompress(packed, len(buf))
            best_d = min(best_d, time.perf_counter() - t)
        out[arm] = (best_c / len(buf), best_d / len(buf))
    return out


def compute_seconds(arm: str, raw_bytes: int) -> float:
    compress, decompress = calibrate()[arm]
    return raw_bytes * (compress + decompress)


def estimate_hints(read_sample: Callable[[], bytes], raw_bytes: int, mbps: float) -> Dict[str, object]:
    """Size estimates and compute times for the codec arms, or none when no codec can win.

    `read_sample` returns the first SAMPLE_BYTES of the payload and is only called when sampling is needed.
    The result carries `compute_s` (seconds per arm) and `stage`, which says where the decision stopped.
    """
    transfer_raw_s = raw_bytes * 8.0 / (mbps * 1e6)
    compute = {arm: compute_seconds(arm, raw_bytes) for arm in CODECS}
    hints: Dict[str, object] = {"estimated": True, "compute_s": {**compute, "raw_blob": 0.0, "text_inline": 0.0},
                                "gzip_bytes": None, "zstd_bytes": None, "brotli_bytes": None}
    if transfer_raw_s <= min(compute.values()):
        hints["stage"] = "break_even_skip"
        return hints
    sample = read_sample()
    keys = {"zstd_blob": "zstd_bytes", "gzip_blob": "gzip_bytes", "brotli_blob": "brotli_bytes"}

    def estimate(arm: str) -> int:
        return int(round(len(CODECS[arm][0](sample)) * raw_bytes / len(sample)))

    zstd_estimate = estimate("zstd_blob")
    zstd_transfer_s = zstd_estimate * 8.0 / (mbps * 1e6)
    if compute["zstd_blob"] + zstd_transfer_s >= transfer_raw_s:
        hints["stage"] = "zstd_not_worth"
        return hints
    hints[keys["zstd_blob"]] = zstd_estimate
    hints["stage"] = "zstd_dominates"
    for arm in SAMPLE_ORDER[1:]:
        if compute[arm] - compute["zstd_blob"] < zstd_transfer_s:
            hints[keys[arm]] = estimate(arm)
            hints["stage"] = "sampled"
    return hints


# ----- Table codec and level ----------------------------------------------------------

TABLE_SAMPLE_ROWS = 20_000
TABLE_SKIP_TRANSFER_S = 0.02  # a default that crosses the link in under this cannot repay the cost of trying the others
TABLE_MIN_GAIN = 0.05  # switch away from the default only for at least this fraction of its estimated cost


def choose_table_arm(sample, total_rows: int, candidates, mbps: float):
    """Pick the cheapest (encode + transfer + decode) table arm from a row sample.

    `candidates` is a list of (arm, encode(table) -> bytes, decode(bytes) -> object, already_encoded); the first is the
    server's default. Sizes and times from the sample are scaled linearly to `total_rows`. An arm whose bytes are already
    cached pays no encode time. Decode time on the server stands in for the client's.
    Returns (arm, report) where report maps arm -> dict(cost_s, bytes, encode_s, decode_s).
    """
    scale = total_rows / max(1, sample.num_rows)
    report: Dict[str, Dict[str, float]] = {}
    for arm, encode, decode, cached in candidates:
        t = time.perf_counter()
        data = encode(sample)
        encode_s = (time.perf_counter() - t) * scale
        t = time.perf_counter()
        decode(data)
        decode_s = (time.perf_counter() - t) * scale
        size = len(data) * scale
        cost = (0.0 if cached else encode_s) + size * 8.0 / (mbps * 1e6) + decode_s
        report[arm] = {"cost_s": cost, "bytes": size, "encode_s": encode_s, "decode_s": decode_s}
    default = candidates[0][0]
    best = min(report, key=lambda a: report[a]["cost_s"])
    if best != default and report[best]["cost_s"] > (1 - TABLE_MIN_GAIN) * report[default]["cost_s"]:
        best = default
    return best, report
