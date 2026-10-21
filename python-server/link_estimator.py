"""Effective link throughput learned from what clients report (experimental, off unless FORMAT_LEARN_LINK=1).

The server cannot see when a client finishes receiving, so the client reports the bytes it downloaded and how long that
took through `record_format_outcome`. The estimate is the median of the most recent rates from downloads large enough that
round trips do not dominate. It is a single estimate for the whole server: it assumes the clients share one link.
"""

from __future__ import annotations

import os
import statistics
from collections import deque
from typing import Deque, Optional

MIN_OBSERVATION_BYTES = 200_000
MIN_OBSERVATIONS = 3
WINDOW = 10

_rates: Deque[float] = deque(maxlen=WINDOW)


def enabled() -> bool:
    return os.environ.get("FORMAT_LEARN_LINK", "").strip().lower() in ("1", "true", "yes")


def observe(num_bytes: Optional[int], seconds: Optional[float]) -> None:
    if num_bytes is None or seconds is None or seconds <= 0 or num_bytes < MIN_OBSERVATION_BYTES:
        return
    _rates.append(num_bytes * 8.0 / seconds / 1e6)


def estimate_mbps() -> Optional[float]:
    """Median recent throughput in Mbit/s, or None until MIN_OBSERVATIONS usable downloads were reported."""
    return statistics.median(_rates) if len(_rates) >= MIN_OBSERVATIONS else None


def reset() -> None:
    _rates.clear()
