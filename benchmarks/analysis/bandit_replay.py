"""Offline replay of arm-selection bandits on the logged unstructured payload runs (results/payloads/*.csv).

Each log row gives the latency of every arm (raw, gzip, zstd, brotli) for one payload on one link, so a bandit's feedback
can be replayed: only the arm it chooses is revealed. This is a replay of single measured latencies, not an online
experiment, and the set is small (418 payload and link pairs: images, speech clips and texts across five links), so it shows how fast
different contexts learn, not how a deployed bandit would do on new traffic.

Policies (epsilon-greedy, epsilon = 0.1, every arm tried once per context before exploiting):
  global       one arm table for everything, the way the existing bandit keys on the target only
  by link      context = link
  by link+kind context = (link, payload kind)
  by link+kind+size  context = (link, kind, size bucket)
  by link, eps 0.01  as "by link" with AdaEdge's online exploration rate
and policies that start from the cost-aware selector's estimate of each arm (compute time plus sampled size over the link
speed, as in cost_model.py) instead of trying every arm once. The bandit then learns only a correction: for each context
and arm, the mean of observed latency / estimated latency, starting from 1 with one pseudo-observation, and the arm
with the lowest estimate times correction wins:
  cost model                      the estimate alone (eps 0, nothing learned)
  cost model + correction (x)     the estimate times a learned correction, context x = global, link or link+kind
  ... link believed 0.25x / 4x    the same with the link speed in the estimate wrong, which the correction must absorb
  cost model, no sampling at 1 Gbit/s or faster   the estimate alone, but those links send raw and skip the sample
  by link, seeded by cost model   the by-link bandit, but the first draw in each link starts every arm at the estimate for that
                                  payload (one pseudo-observation) instead of trying each arm once, and sampling is paid only then
Every estimate needs a 256 KB sample of the payload, so its measured cost is added to the latency of every draw of the
cost-model policies, and once per link for the seeded ones.
References: the best single arm in hindsight, the best arm per link in hindsight, the per-step oracle, and the
selector's own logged choice.

    .venv/bin/python -m benchmarks.analysis.bandit_replay
"""

from __future__ import annotations

import csv
import math
import random
import statistics as st
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

from benchmarks.analysis import cost_model

ARMS = ["raw", "gzip", "zstd", "brotli"]
LINKS = ["localhost", "lan", "wan", "cellular", "badwifi"]
STEPS, SEEDS, EPSILON = 2000, 20, 0.1
CHECKPOINTS = (100, 500, 2000)


def load_items():
    items = []
    for link in LINKS:
        path = Path("results/payloads") / f"{link}.csv"
        by = defaultdict(dict)
        for r in csv.DictReader(open(path)):
            by[r["payload"]][r["strategy"]] = r
        for name, v in by.items():
            if not all(a in v for a in ARMS):
                continue
            items.append({"link": link, "payload": name, "kind": v["raw"]["kind"], "size": int(v["raw"]["raw_bytes"]),
                          "lat": {a: float(v[a]["latency_s"]) for a in ARMS},
                          "selector": float(v["selector min_latency"]["latency_s"])})
    return items


def size_bucket(n):
    return int(math.log2(max(n, 1)) // 3)


CONTEXTS = {
    "global": lambda it: (),
    "by link": lambda it: (it["link"],),
    "by link+kind": lambda it: (it["link"], it["kind"]),
    "by link+kind+size": lambda it: (it["link"], it["kind"], size_bucket(it["size"])),
}


LINK_MBPS = {link: mbps or cost_model.LOOPBACK_MBPS for link, mbps in cost_model.LINKS}
BELIEFS = (0.25, 4.0)


def estimates(items):
    """Per item: {belief: {arm: estimated seconds}} and the seconds the 256 KB sample cost.

    The estimate is the cost-aware selector's: compute(arm, size) fitted on the other payloads at localhost, plus the
    arm's sampled wire size over the link speed (times `belief`, for the wrong-link-speed variants).
    """
    sampled = cost_model.estimate_all(Namespace(images="data/m3bench/images", audio="data/audio",
                                                bird_root="data/dev_20240627", image_stride=1))
    localhost = cost_model.load("localhost")
    models = {}
    out = []
    for it in items:
        name = it["payload"]
        if name not in models:
            models[name] = cost_model.fit_compute(localhost, name)
        wire, spent, _ = sampled[name]
        speed = LINK_MBPS[it["link"]]
        per_belief = {}
        for belief in (1.0,) + BELIEFS:
            per_belief[belief] = {a: max(models[name][a][0] + models[name][a][1] * it["size"]
                                         + wire[a] * 8 / (speed * belief * 1e6), 1e-6) for a in ARMS}
        out.append((per_belief, spent))
    return out


def run_prior_policy(items, est, stream, context, rng, belief, learn, epsilon=EPSILON, skip_fast=False):
    """Pick min(estimate x learned correction); epsilon-greedy over the arms; the sample's cost is paid on every draw."""
    ratio_sum, ratio_n = defaultdict(float), defaultdict(int)
    chosen_total, relative_total, trace = 0.0, 0.0, []
    for step, i in enumerate(stream, 1):
        it, (per_belief, spent) = items[i], est[i]
        prior = per_belief[belief]
        c = context(it)
        if skip_fast and LINK_MBPS[it["link"]] >= cost_model.FAST_LINK_MBPS:
            spent = 0.0

        def correction(a):
            n = ratio_n[(c, a)]
            return (1.0 + ratio_sum[(c, a)]) / (1 + n)

        if learn and rng.random() < epsilon:
            arm = rng.choice(ARMS)
        else:
            arm = "raw" if spent == 0.0 and skip_fast else min(ARMS, key=lambda a: prior[a] * (correction(a) if learn else 1.0))
        latency = it["lat"][arm]
        if learn:
            ratio_sum[(c, arm)] += latency / prior[arm]
            ratio_n[(c, arm)] += 1
        paid = latency + spent
        chosen_total += paid
        relative_total += paid / min(it["lat"].values())
        if step in CHECKPOINTS:
            trace.append((chosen_total, relative_total / step))
    return trace


def run_policy(items, stream, context, rng, epsilon=EPSILON, seed=None):
    """Epsilon-greedy on the mean latency per (context, arm). With seed=(est, belief), the first draw in a context starts
    every arm at the cost model's estimate for that payload (one pseudo-observation) instead of trying each arm once, and
    pays for that payload's 256 KB sample once."""
    sums, counts = defaultdict(float), defaultdict(int)
    seeded, chosen_total, relative_total, trace = set(), 0.0, 0.0, []
    for step, i in enumerate(stream, 1):
        it = items[i]
        c = context(it)
        sample_cost = 0.0
        if seed is not None and c not in seeded:
            seeded.add(c)
            per_belief, sample_cost = seed[0][i]
            for a in ARMS:
                sums[(c, a)], counts[(c, a)] = per_belief[seed[1]][a], 1
        untried = [a for a in ARMS if counts[(c, a)] == 0]
        if untried:
            arm = rng.choice(untried)
        elif rng.random() < epsilon:
            arm = rng.choice(ARMS)
        else:
            arm = min(ARMS, key=lambda a: sums[(c, a)] / counts[(c, a)])
        latency = it["lat"][arm]
        sums[(c, arm)] += latency
        counts[(c, arm)] += 1
        chosen_total += latency + sample_cost
        relative_total += (latency + sample_cost) / min(it["lat"].values())
        if step in CHECKPOINTS:
            trace.append((chosen_total, relative_total / step))
    return trace


PRIOR_POLICIES = [
    ("cost model", (), 1.0, False),
    ("cost model + by-link correction", ("link",), 1.0, True),
    ("cost model, link believed 0.25x", ("link",), 0.25, False),
    ("cost model + by-link correction, link believed 0.25x", ("link",), 0.25, True),
    ("cost model, link believed 4x", ("link",), 4.0, False),
    ("cost model + by-link correction, link believed 4x", ("link",), 4.0, True),
]


def main():
    items = load_items()
    est = estimates(items)
    rows, relative = defaultdict(list), defaultdict(list)
    for seed in range(SEEDS):
        rng = random.Random(seed)
        stream = [rng.randrange(len(items)) for _ in range(STEPS)]
        oracle = [sum(min(items[i]["lat"].values()) for i in stream[:n]) for n in CHECKPOINTS]
        rel = lambda arm_of, n: st.mean(items[i]["lat"][arm_of(items[i])] / min(items[i]["lat"].values()) for i in stream[:n])
        for name, ctx in CONTEXTS.items():
            trace = run_policy(items, stream, ctx, random.Random(seed + 1000))
            rows[name].append([t / o for (t, _), o in zip(trace, oracle)])
            relative[name].append([r for _, r in trace])
        trace = run_policy(items, stream, CONTEXTS["by link"], random.Random(seed + 1000), epsilon=0.01)
        rows["by link, eps 0.01"].append([t / o for (t, _), o in zip(trace, oracle)])
        relative["by link, eps 0.01"].append([r for _, r in trace])
        for name, belief in (("by link, seeded by cost model", 1.0), ("by link, seeded, link believed 0.25x", 0.25),
                             ("by link, seeded, link believed 4x", 4.0)):
            trace = run_policy(items, stream, CONTEXTS["by link"], random.Random(seed + 1000), seed=(est, belief))
            rows[name].append([t / o for (t, _), o in zip(trace, oracle)])
            relative[name].append([r for _, r in trace])
        trace = run_prior_policy(items, est, stream, lambda it: (), random.Random(seed + 1000), 1.0, False, skip_fast=True)
        rows["cost model, no sampling at 1 Gbit/s or faster"].append([t / o for (t, _), o in zip(trace, oracle)])
        relative["cost model, no sampling at 1 Gbit/s or faster"].append([r for _, r in trace])
        for name, keys, belief, learn in PRIOR_POLICIES:
            ctx = lambda it, keys=keys: tuple(it[k] for k in keys)
            trace = run_prior_policy(items, est, stream, ctx, random.Random(seed + 1000), belief, learn)
            rows[name].append([t / o for (t, _), o in zip(trace, oracle)])
            relative[name].append([r for _, r in trace])
        best_global = min(ARMS, key=lambda a: sum(items[i]["lat"][a] for i in stream))
        rows["best single arm (hindsight)"].append([sum(items[i]["lat"][best_global] for i in stream[:n]) / o for n, o in zip(CHECKPOINTS, oracle)])
        relative["best single arm (hindsight)"].append([rel(lambda it: best_global, n) for n in CHECKPOINTS])
        per_link = {l: min(ARMS, key=lambda a: sum(items[i]["lat"][a] for i in stream if items[i]["link"] == l)) for l in LINKS
                    if any(items[i]["link"] == l for i in stream)}
        rows["best arm per link (hindsight)"].append([sum(items[i]["lat"][per_link[items[i]["link"]]] for i in stream[:n]) / o for n, o in zip(CHECKPOINTS, oracle)])
        rows["selector (logged choice)"].append([sum(items[i]["selector"] for i in stream[:n]) / o for n, o in zip(CHECKPOINTS, oracle)])
        relative["selector (logged choice)"].append([st.mean(items[i]["selector"] / min(items[i]["lat"].values()) for i in stream[:n]) for n in CHECKPOINTS])

    lines = ["# Bandit replay on the logged payload runs", "",
             f"{len(items)} (link, payload) items; {SEEDS} random streams of {STEPS} draws; total latency / per-step oracle "
             "(1.00x is perfect), mean over streams. Cumulative, so the first columns include the learning cost.", "",
             "| policy | " + " | ".join(f"first {n} draws" for n in CHECKPOINTS) + " |", "|---|" + "---|" * len(CHECKPOINTS)]
    for name, vals in rows.items():
        lines.append(f"| {name} | " + " | ".join(f"{st.mean(v[k] for v in vals):.2f}x" for k in range(len(CHECKPOINTS))) + " |")
    lines += ["", "Same replay, but each draw weighted equally instead of by its latency (mean over draws of the chosen arm's latency / "
              "the best arm's latency), so the fast links count as much as the slow ones:", "",
              "| policy | " + " | ".join(f"first {n} draws" for n in CHECKPOINTS) + " |", "|---|" + "---|" * len(CHECKPOINTS)]
    for name, vals in relative.items():
        lines.append(f"| {name} | " + " | ".join(f"{st.mean(v[k] for v in vals):.2f}x" for k in range(len(CHECKPOINTS))) + " |")
    text = "\n".join(lines) + "\n"
    Path("results/payloads/bandit_replay.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
