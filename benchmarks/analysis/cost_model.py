"""Offline replay of a cost-aware selector on the unstructured payload runs (results/payloads/).

The current selector picks the arm with the fewest bytes (or, with a network speed set, the least transfer time) and
ignores how long compressing and decompressing take, which is why it loses on fast links. The policy tested here scores
each arm as  compute(arm, size) + wire_bytes / bandwidth  and takes the minimum.

Replay rules: the compute model is fitted on the *other* payloads (leave one out) and the chosen arm's latency is the
one measured when that arm was forced on that payload and link. The policy only chooses among the four arms measured
(raw, gzip, zstd, brotli). Knowing each arm's size costs something, so three variants are scored:

  free exact sizes   the sizes the server's hints report, at no cost (optimistic; not achievable)
  exact + hints      the same sizes plus the measured hint cost: the selector rows' select_s minus the fastest select_s
                     on that link (the bare tool-call round trip); a lower bound on the overhead
  sampled sizes      sizes extrapolated from compressing only the first 256 KB at the server's codec levels, plus the
                     measured time that took
  sampled + fast skip  the sampled policy, except that a link of 1 Gbit/s or more skips sampling and sends raw

The sampled variant is also rerun with the link speed believed to be 0.25x and 4x the true one.

    .venv/bin/python -m benchmarks.analysis.cost_model
"""

from __future__ import annotations

import csv
import gzip
import time
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa

from benchmarks.sweeps.payloads import all_payloads

R = Path("results/payloads")
ARMS = ["raw", "gzip", "zstd", "brotli"]
LINKS = [("localhost", None), ("lan", 1000.0), ("wan", 50.0), ("cellular", 10.0), ("badwifi", 5.0)]
LOOPBACK_MBPS = 20000.0  # localhost has no emulated limit; treat it as a very fast link


def load(link):
    out = defaultdict(dict)
    with open(R / f"{link}.csv") as f:
        for r in csv.DictReader(f):
            out[r["payload"]][r["strategy"]] = r
    return out


def fit_compute(localhost, exclude):
    """Per arm, compute seconds = a + b * raw_bytes, from localhost latencies of every payload except *exclude*."""
    models = {}
    for arm in ARMS:
        x = np.array([int(v[arm]["raw_bytes"]) for p, v in localhost.items() if p != exclude])
        y = np.array([float(v[arm]["latency_s"]) for p, v in localhost.items() if p != exclude])
        b, a = np.polyfit(x, y, 1)
        models[arm] = (max(a, 0.0), max(b, 0.0))
    return models


SAMPLE_BYTES = 256 * 1024
SAMPLE_CODECS = {"gzip": lambda b: gzip.compress(b), "zstd": lambda b: pa.Codec("zstd", compression_level=3).compress(b, asbytes=True),
                 "brotli": lambda b: pa.Codec("brotli", compression_level=5).compress(b, asbytes=True)}  # the server's levels
FAST_LINK_MBPS = 1000.0
BANDWIDTH_ERRORS = (0.25, 1.0, 4.0)


def sampled_estimates(data):
    """Wire-size estimate per arm from compressing only the first 256 KB, and what taking the sample cost in seconds."""
    sample = data[:SAMPLE_BYTES]
    wire = {"raw": len(data)}
    t = time.perf_counter()
    for arm, compress in SAMPLE_CODECS.items():
        wire[arm] = len(compress(sample)) * len(data) / len(sample)
    return wire, time.perf_counter() - t


def estimate_all(args):
    out = {}
    for name, _, data, _ in all_payloads(args):
        out[name] = sampled_estimates(data)
    return out


def choose(models, per_arm_wire, raw_bytes, mbps):
    best, best_cost = None, float("inf")
    for arm in ARMS:
        a, b = models[arm]
        cost = a + b * raw_bytes + per_arm_wire[arm] * 8 / (mbps * 1e6)
        if cost < best_cost:
            best, best_cost = arm, cost
    return best


def policy_totals(data, localhost, payloads, speed, variant, belief=1.0):
    total, picks = 0.0, defaultdict(int)
    floor = min(float(data[p]["selector min_bytes"]["select_s"]) for p in data)
    for p in payloads:
        v = data[p]
        raw_bytes = int(v["raw"]["raw_bytes"])
        exact = {a: int(v[a]["wire_bytes"]) for a in ARMS}
        extra = 0.0
        if variant == "exact+hints":
            extra = max(float(v["selector min_bytes"]["select_s"]) - floor, 0.0)
            sizes = exact
        elif variant == "sampled":
            sizes, extra = ESTIMATES[p]
        elif variant == "sampled+fast skip":
            if speed >= FAST_LINK_MBPS:
                total += float(v["raw"]["latency_s"])
                picks["raw"] += 1
                continue
            sizes, extra = ESTIMATES[p]
        else:
            sizes = exact
        pick = choose(fit_compute(localhost, p), sizes, raw_bytes, speed * belief)
        picks[pick] += 1
        total += float(v[pick]["latency_s"]) + extra
    return total, picks


def main():
    global ESTIMATES
    ESTIMATES = estimate_all(Namespace(images="data/m3bench/images", audio="data/audio", bird_root="data/dev_20240627",
                                       image_stride=1))
    localhost = load("localhost")
    rows, belief_rows = [], []
    for link, mbps in LINKS:
        data = load(link)
        speed = mbps or LOOPBACK_MBPS
        for kind in ("image", "audio", "text"):
            payloads = [p for p, v in data.items() if v["raw"]["kind"] == kind]
            totals = {"oracle": sum(min(float(data[p][a]["latency_s"]) for a in ARMS) for p in payloads)}
            for a in ARMS:
                totals[f"fixed {a}"] = sum(float(data[p][a]["latency_s"]) for p in payloads)
            for target in ("min_bytes", "min_latency"):
                totals[f"selector {target}"] = sum(float(data[p][f"selector {target}"]["latency_s"]) for p in payloads)
            picks = {}
            for variant in ("free exact sizes", "exact+hints", "sampled", "sampled+fast skip"):
                totals[variant], picks[variant] = policy_totals(data, localhost, payloads, speed, variant)
            for belief in BANDWIDTH_ERRORS:
                t, _ = policy_totals(data, localhost, payloads, speed, "sampled", belief)
                belief_rows.append((link, kind, belief, t / totals["oracle"]))
            rows.append((link, kind, len(payloads), totals, picks))

    lines = ["# Cost-aware selector, offline replay", "",
             "Latency totals divided by the per-payload oracle (the best of the four arms, measured). 1.00x is perfect. "
             "`selector` rows are what the current server selector did in the run, hint computation included. "
             "See the module docstring for what each cost-aware variant is charged.", ""]
    cols = ["free exact sizes", "exact+hints", "sampled", "sampled+fast skip", "selector min_bytes", "selector min_latency"]
    lines += ["| link | kind (n) | oracle total (s) | " + " | ".join(cols) + " | best fixed arm |", "|" + "---|" * (len(cols) + 4)]
    for link, kind, n, totals, _ in rows:
        o = totals["oracle"]
        best = min(ARMS, key=lambda a: totals[f"fixed {a}"])
        lines.append(f"| {link} | {kind} ({n}) | {o:.2f} | " + " | ".join(f"{totals[c] / o:.2f}x" for c in cols)
                     + f" | {totals['fixed ' + best] / o:.2f}x ({best}) |")

    lines += ["", "## Arms picked by the sampled variant", "", "| link | kind | picks |", "|---|---|---|"]
    for link, kind, _, _, picks in rows:
        lines.append(f"| {link} | {kind} | " + ", ".join(f"{k}: {v}" for k, v in sorted(picks["sampled"].items())) + " |")

    lines += ["", "## Sampled variant when the believed link speed is wrong (regret vs oracle)", "",
              "| link | kind | believed 0.25x | believed 1x | believed 4x |", "|---|---|---|---|---|"]
    by = defaultdict(dict)
    for link, kind, belief, r in belief_rows:
        by[(link, kind)][belief] = r
    for (link, kind), d in by.items():
        lines.append(f"| {link} | {kind} | " + " | ".join(f"{d[b]:.2f}x" for b in BANDWIDTH_ERRORS) + " |")
    out = "\n".join(lines) + "\n"
    (R / "cost_model.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
