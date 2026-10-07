"""
Multi-armed bandit (MAB) for format selection: reward estimates, epsilon-greedy selection,
outcome recording, and persistence. AdaEdge-inspired; see IMPLEMENTATION_PLAN_format_selection.md.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from format_selector import (
    FormatName,
    HintsDict,
    OptimizationTarget,
    SelectionContext,
    select_format,
    select_format_with_hints,
)

# Outcome record: context + choice + observed metrics (for history and reward)
OutcomeDict = Dict[str, Any]  # n_rows, n_cols, target, format, bytes, latency_s, time_to_first_rows_s?

# MAB state: per (target, format) -> {sum_reward, count}
MABStateDict = Dict[str, Dict[str, Dict[str, float]]]  # target -> format -> {sum_reward, count}

DEFAULT_HISTORY_PATH = Path("results/format_selection_history.jsonl")
DEFAULT_MAB_STATE_PATH = Path("results/format_mab_state.json")
FORMATS: tuple[FormatName, ...] = (
    "json",
    "parquet_blob",
    "parquet_stream",
    "arrow_ipc_blob",
    "arrow_ipc_stream",
    "text_inline",
    "raw_blob",
    "gzip_blob",
)


def reward_from_outcome(outcome: OutcomeDict, target: OptimizationTarget) -> float:
    """
    Compute reward from outcome (to maximize). Higher is better.
    min_bytes -> -bytes; min_latency -> -latency_s; min_time_to_first_rows -> -time_to_first_rows_s.
    """
    if target == OptimizationTarget.MIN_BYTES:
        b = outcome.get("bytes")
        if b is None:
            return 0.0
        return -float(b)
    if target == OptimizationTarget.MIN_LATENCY:
        s = outcome.get("latency_s")
        if s is None:
            return 0.0
        return -float(s)
    if target == OptimizationTarget.MIN_TIME_TO_FIRST_ROWS:
        ttfr = outcome.get("time_to_first_rows_s")
        if ttfr is not None:
            return -float(ttfr)
        # Fallback: use latency as proxy
        s = outcome.get("latency_s")
        if s is not None:
            return -float(s)
        return 0.0
    return 0.0


def _state_key(target: OptimizationTarget, format_name: FormatName) -> tuple[str, str]:
    return (target.value, format_name)


def load_mab_state(path: Optional[os.PathLike[str]] = None) -> MABStateDict:
    """Load MAB state from JSON; return empty dict if file missing or invalid."""
    p = Path(path) if path is not None else DEFAULT_MAB_STATE_PATH
    if not p.is_file():
        return {}
    try:
        with open(p) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    # Normalize: data is {target: {format: {sum_reward, count}}}
    out: MABStateDict = {}
    for target_val, formats_dict in data.items():
        if not isinstance(formats_dict, dict):
            continue
        out[target_val] = {}
        for fmt, stats in formats_dict.items():
            if isinstance(stats, dict) and "sum_reward" in stats and "count" in stats:
                out[target_val][fmt] = {
                    "sum_reward": float(stats["sum_reward"]),
                    "count": float(stats["count"]),
                }
    return out


def save_mab_state(state: MABStateDict, path: Optional[os.PathLike[str]] = None) -> None:
    """Persist MAB state to JSON."""
    p = Path(path) if path is not None else DEFAULT_MAB_STATE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(state, f, indent=2)


def select_format_with_mab(
    context: SelectionContext,
    hints: Optional[HintsDict],
    mab_state: Optional[MABStateDict],
    *,
    epsilon: Optional[float] = None,
) -> FormatName:
    """
    Epsilon-greedy format selection using MAB state. If mab_state is None or has no
    data for this target, fall back to select_format_with_hints (or select_format).
    """
    if mab_state is None:
        if hints:
            return select_format_with_hints(context, hints)
        return select_format(context)

    target_val = context.target.value
    state_for_target = mab_state.get(target_val)
    if not state_for_target:
        if hints:
            return select_format_with_hints(context, hints)
        return select_format(context)

    # Check if we have any observations
    has_any = any(
        state_for_target.get(fmt, {}).get("count", 0) > 0 for fmt in FORMATS
    )
    if not has_any:
        if hints:
            return select_format_with_hints(context, hints)
        return select_format(context)

    eps = epsilon if epsilon is not None else float(os.environ.get("FORMAT_SELECT_EPSILON", "0.1"))
    if random.random() < eps:
        return random.choice(list(FORMATS))

    # Greedy: argmax Q(format) = sum_reward / count
    best_format: Optional[FormatName] = None
    best_q = float("-inf")
    for fmt in FORMATS:
        stats = state_for_target.get(fmt, {})
        count = stats.get("count", 0) or 0
        if count <= 0:
            continue
        q = (stats.get("sum_reward", 0) or 0) / count
        if q > best_q:
            best_q = q
            best_format = fmt

    if best_format is not None:
        return best_format
    if hints:
        return select_format_with_hints(context, hints)
    return select_format(context)


def record_outcome(
    context: SelectionContext,
    format_used: FormatName,
    outcome: OutcomeDict,
    mab_state: Optional[MABStateDict] = None,
    history_path: Optional[os.PathLike[str]] = None,
) -> None:
    """
    Append outcome to history file and update MAB state (sum_reward, count) for (target, format_used).
    """
    # Build record for history
    record: OutcomeDict = {
        "n_rows": context.n_rows,
        "n_cols": context.n_cols,
        "target": context.target.value,
        "format": format_used,
        "bytes": outcome.get("bytes"),
        "latency_s": outcome.get("latency_s"),
        "time_to_first_rows_s": outcome.get("time_to_first_rows_s"),
    }

    # Append to history file
    p = Path(history_path) if history_path is not None else DEFAULT_HISTORY_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(record) + "\n")

    # Update MAB state
    if mab_state is not None:
        reward = reward_from_outcome(outcome, context.target)
        target_val = context.target.value
        if target_val not in mab_state:
            mab_state[target_val] = {}
        if format_used not in mab_state[target_val]:
            mab_state[target_val][format_used] = {"sum_reward": 0.0, "count": 0.0}
        mab_state[target_val][format_used]["sum_reward"] += reward
        mab_state[target_val][format_used]["count"] += 1.0


def mab_enabled() -> bool:
    """True if FORMAT_SELECT_MAB env is set to enable MAB selection."""
    return os.environ.get("FORMAT_SELECT_MAB", "").strip() in ("1", "true", "yes")
