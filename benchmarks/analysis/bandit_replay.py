"""Offline replay of arm-selection bandits on the logged unstructured payload runs (results/payloads/*.csv).

Each log row gives the latency of every arm (raw, gzip, zstd, brotli) for one payload on one link, so a bandit's feedback
can be replayed: only the arm it chooses is revealed. This is a replay of single measured latencies, not an online
experiment, and the payload set is small (258 images, audio clips and texts across five links), so it shows how fast
different contexts learn, not how a deployed bandit would do on new traffic.

Policies (epsilon-greedy, epsilon = 0.1, every arm tried once per context before exploiting):
  global       one arm table for everything, the way the existing bandit keys on the target only
  by link      context = link
  by link+kind context = (link, payload kind)
  by link+kind+size  context = (link, kind, size bucket)
and references: the best single arm in hindsight, the best arm per link in hindsight, the per-step oracle, and the
selector's own logged choice.

    .venv/bin/python -m benchmarks.analysis.bandit_replay
"""

from __future__ import annotations

import csv
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

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
            items.append({"link": link, "kind": v["raw"]["kind"], "size": int(v["raw"]["raw_bytes"]),
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


def run_policy(items, stream, context, rng):
    sums, counts = defaultdict(float), defaultdict(int)
    chosen_total, relative_total, trace = 0.0, 0.0, []
    for step, i in enumerate(stream, 1):
        it = items[i]
        c = context(it)
        untried = [a for a in ARMS if counts[(c, a)] == 0]
        if untried:
            arm = rng.choice(untried)
        elif rng.random() < EPSILON:
            arm = rng.choice(ARMS)
        else:
            arm = min(ARMS, key=lambda a: sums[(c, a)] / counts[(c, a)])
        latency = it["lat"][arm]
        sums[(c, arm)] += latency
        counts[(c, arm)] += 1
        chosen_total += latency
        relative_total += latency / min(it["lat"].values())
        if step in CHECKPOINTS:
            trace.append((chosen_total, relative_total / step))
    return trace


def main():
    items = load_items()
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
