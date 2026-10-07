"""Summarize the {java,python} x {default,data_driven} BIRD matrix into markdown tables."""

from __future__ import annotations

import csv
from pathlib import Path

RESULTS = Path("results")
CONFIGS = [(i, e) for i in ("python", "java") for e in ("data_driven", "default")]


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    data = {}
    for impl, enc in CONFIGS:
        static = {r["question_id"]: r for r in load(RESULTS / f"bird/matrix_{impl}_{enc}_static.csv")}
        sel = {
            (r["question_id"], r["target"]): r
            for r in load(RESULTS / f"bird/matrix_{impl}_{enc}_selector.csv")
        }
        data[(impl, enc)] = (static, sel)

    common = set.intersection(*(set(s) for s, _ in data.values()))
    print(f"queries completed by all four configs: {len(common)}\n")

    rows = []
    for (impl, enc), (static, sel) in data.items():
        mb = lambda col: sum(float(static[q][col]) for q in common) / 1e6
        sel_b = sum(float(sel[(q, "min_bytes")]["bytes"]) for q in common) / 1e6
        rows.append(
            [impl, enc, f"{mb('json_bytes'):.2f}", f"{mb('parquet_bytes'):.2f}",
             f"{mb('arrow_ipc_bytes'):.2f}", f"{sel_b:.2f}",
             f"{mb('json_bytes') / sel_b:.1f}x"]
        )
    print("### Wire bytes on the common query set (MB)\n")
    print(table(rows, ["server", "parquet encoding", "always JSON", "always Parquet",
                       "always Arrow IPC", "selector (min_bytes)", "selector vs JSON"]))

    rows = []
    for (impl, enc), (static, sel) in data.items():
        tot = lambda col: sum(float(static[q][col]) for q in common)
        sel_l = sum(float(sel[(q, "min_latency")]["latency_s"]) for q in common)
        sel_bl = sum(float(sel[(q, "min_bytes")]["latency_s"]) for q in common)
        rows.append([impl, enc, f"{tot('json_s'):.1f}", f"{tot('parquet_s'):.1f}",
                     f"{tot('arrow_ipc_s'):.1f}", f"{sel_bl:.1f}", f"{sel_l:.1f}"])
    print("\n### Total localhost latency on the common set (s; SQL execution included)\n")
    print(table(rows, ["server", "parquet encoding", "always JSON", "always Parquet",
                       "always Arrow IPC", "selector min_bytes", "selector min_latency"]))

    print("\n### Per-query JSON byte agreement between servers (same encoding)\n")
    rows = []
    for enc in ("data_driven", "default"):
        py, jv = data[("python", enc)][0], data[("java", enc)][0]
        same = sum(py[q]["json_bytes"] == jv[q]["json_bytes"] for q in common)
        rows.append([enc, f"{same}/{len(common)}"])
    print(table(rows, ["parquet encoding", "identical JSON bytes"]))

    print("\n### Format chosen by the selector (min_bytes)\n")
    rows = []
    for (impl, enc), (_, sel) in data.items():
        counts = {}
        for q in common:
            f = sel[(q, "min_bytes")]["chosen_format"]
            counts[f] = counts.get(f, 0) + 1
        rows.append([impl, enc, ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))])
    print(table(rows, ["server", "parquet encoding", "choices"]))


if __name__ == "__main__":
    main()
