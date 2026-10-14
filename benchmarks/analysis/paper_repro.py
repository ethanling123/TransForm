"""Compare a rerun of the workshop paper's harness against the paper's own raw records.

The paper's per-query records (results/bird_e2e_dev_full_gold_round2_{both,server,inline}.jsonl) only exist in git
history (commit 55de98c); they are read with `git show`. The rerun comes from benchmarks/scripts/run_paper_repro.sh.

    .venv/bin/python -m benchmarks.analysis.paper_repro --run results/paper_repro/run1 --out results/paper_repro/run1/comparison.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from collections import Counter
from pathlib import Path

PAPER_COMMIT = "55de98c"
PAPER_FILE = "results/bird_e2e_dev_full_gold_round2_{}.jsonl"


def records(lines):
    rows = [json.loads(line) for line in lines if line.strip()]
    return [r for r in rows if r.get("question_id") is not None and "baseline_bytes" in r]


def paper_records(arm):
    text = subprocess.run(["git", "show", f"{PAPER_COMMIT}:{PAPER_FILE.format(arm)}"], capture_output=True, text=True,
                          check=True).stdout
    return records(text.splitlines())


def rerun_records(run_dir, arm):
    return records(Path(run_dir, f"{arm}.jsonl").read_text().splitlines())


def total(rows, *keys):
    return sum(r[k] for r in rows for k in keys if r.get(k) is not None)


def per_query(rows, *keys):
    return [sum(r[k] for k in keys if r.get(k) is not None) for r in rows]


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def summarize(both, server):
    """Aggregates in the paper's terms: baseline (JSON), client-driven (describe + fetch), TransForm (fused call)."""
    arms = {
        "Baseline (JSON)": (both, ("baseline_fetch_s",), "baseline_bytes"),
        "Client-driven": (both, ("describe_s", "enhanced_fetch_s"), "enhanced_bytes"),
        "TransForm (fused)": (server, ("server_auto_call_s", "server_auto_payload_s"), "server_auto_bytes"),
    }
    out = {}
    for name, (rows, time_keys, byte_key) in arms.items():
        transport = per_query(rows, *time_keys)
        register = per_query(rows, "register_s")
        out[name] = {
            "n": len(rows),
            "bytes": total(rows, byte_key),
            "transport_s": sum(transport),
            "register_plus_transport_s": sum(transport) + sum(register),
            "median_ms": 1000 * statistics.median(transport),
            "p95_ms": 1000 * pct(transport, 0.95),
            "p99_ms": 1000 * pct(transport, 0.99),
            "median_bytes": statistics.median(r[byte_key] for r in rows if r.get(byte_key) is not None),
        }
    out["formats"] = {
        "client-driven": dict(Counter(r["recommended_format"] for r in both)),
        "fused": dict(Counter(r["server_auto_chosen_format"] for r in server)),
    }
    return out


def per_query_agreement(paper_both, paper_server, new_both, new_server):
    """Do the two runs agree query by query on the choice and the bytes?"""
    def by_id(rows):
        return {r["question_id"]: r for r in rows}
    pb, ps, nb, ns = map(by_id, (paper_both, paper_server, new_both, new_server))
    common = sorted(set(pb) & set(nb) & set(ps) & set(ns))
    same_baseline = sum(pb[q]["baseline_bytes"] == nb[q]["baseline_bytes"] for q in common)
    same_choice = sum(ps[q]["server_auto_chosen_format"] == ns[q]["server_auto_chosen_format"] for q in common)
    same_wire = sum(ps[q]["server_auto_bytes"] == ns[q]["server_auto_bytes"] for q in common)
    differing = [q for q in common if pb[q]["baseline_bytes"] != nb[q]["baseline_bytes"]]
    return {"queries": len(common), "same_baseline_bytes": same_baseline, "same_chosen_format": same_choice,
            "same_wire_bytes": same_wire, "differing_baseline_ids": differing[:10]}


def mb(x):
    return f"{x / 1e6:.2f} MB"


def table(title, paper, new):
    lines = [f"### {title}", "", "| Arm | Paper-run sum bytes | Rerun sum bytes | Paper-run register+transport (s) | "
             "Rerun register+transport (s) | Paper-run transport only (s) | Rerun transport only (s) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ("Baseline (JSON)", "Client-driven", "TransForm (fused)"):
        p, n = paper[arm], new[arm]
        lines.append(f"| {arm} | {mb(p['bytes'])} | {mb(n['bytes'])} | {p['register_plus_transport_s']:.1f} | "
                     f"{n['register_plus_transport_s']:.1f} | {p['transport_s']:.1f} | {n['transport_s']:.1f} |")
    lines += ["", "| Arm | Paper-run median / p95 / p99 (ms) | Rerun median / p95 / p99 (ms) |", "|---|---|---|"]
    for arm in ("Baseline (JSON)", "Client-driven", "TransForm (fused)"):
        p, n = paper[arm], new[arm]
        lines.append(f"| {arm} | {p['median_ms']:.1f} / {p['p95_ms']:.1f} / {p['p99_ms']:.1f} | "
                     f"{n['median_ms']:.1f} / {n['p95_ms']:.1f} / {n['p99_ms']:.1f} |")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="results/paper_repro/run1")
    parser.add_argument("--java", default="results/paper_repro/java", help="rerun on the Java server (the paper's server)")
    parser.add_argument("--default-encoding", default="results/paper_repro/run1_default_encoding")
    parser.add_argument("--out", default="results/paper_repro/comparison.md")
    args = parser.parse_args()

    paper_both, paper_server = paper_records("both"), paper_records("server")
    new_both, new_server = rerun_records(args.run, "both"), rerun_records(args.run, "server")
    paper = summarize(paper_both, paper_server)
    new = summarize(new_both, new_server)
    agreement = per_query_agreement(paper_both, paper_server, new_both, new_server)

    lines = ["# Paper harness rerun against the Python server", "",
             f"Paper numbers come from the raw records in commit {PAPER_COMMIT}; the rerun is "
             f"`benchmarks/paper_harness/bench_bird_e2e.py` (unmodified) against `python-server/`.", ""]
    lines += table("Aggregates over the full BIRD dev set (1534 gold-SQL queries)", paper, new)
    lines += ["", "### Format choices", "", f"- Paper-run: client-driven {paper['formats']['client-driven']}, "
              f"fused {paper['formats']['fused']}", f"- Rerun: client-driven {new['formats']['client-driven']}, "
              f"fused {new['formats']['fused']}", "", "### Query-by-query agreement", "",
              f"- {agreement['queries']} queries present in all four runs",
              f"- identical baseline JSON bytes: {agreement['same_baseline_bytes']}",
              f"- identical fused-arm chosen format: {agreement['same_chosen_format']}",
              f"- identical fused-arm wire bytes: {agreement['same_wire_bytes']}",
              f"- first queries whose baseline bytes differ: {agreement['differing_baseline_ids']}"]
    if Path(args.default_encoding, "server.jsonl").exists():
        dflt = rerun_records(args.default_encoding, "server")
        same = sum(a["server_auto_bytes"] == b["server_auto_bytes"] for a, b in
                   zip(sorted(paper_server, key=lambda r: r["question_id"]), sorted(dflt, key=lambda r: r["question_id"])))
        lines += ["", "### Parquet encoding strategy (fused arm, Python server)", "",
                  f"- Rerun with `PARQUET_ENCODING_STRATEGY=default`: {mb(total(dflt, 'server_auto_bytes'))}, "
                  f"{same} of {len(dflt)} queries have wire bytes identical to the paper-run "
                  f"({mb(total(paper_server, 'server_auto_bytes'))}).",
                  f"- With `data_driven` (the runbook's setting): {mb(new['TransForm (fused)']['bytes'])}. "
                  "The paper-run used the default encoding."]
    if Path(args.java, "server.jsonl").exists():
        java = summarize(rerun_records(args.java, "both"), rerun_records(args.java, "server"))
        base, client, fused = (java[a] for a in ("Baseline (JSON)", "Client-driven", "TransForm (fused)"))
        lines += ["", "## Java server (the paper's headline table)", "",
                  "| Arm | Sum bytes | Transport only (s) | vs baseline | Register + transport (s) | vs baseline |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name, a in (("Baseline (JSON)", base), ("Client-driven", client), ("TransForm (fused)", fused)):
            lines.append(f"| {name} | {mb(a['bytes'])} | {a['transport_s']:.1f} | "
                         f"{a['transport_s'] / base['transport_s'] - 1:+.0%} | {a['register_plus_transport_s']:.1f} | "
                         f"{a['register_plus_transport_s'] / base['register_plus_transport_s'] - 1:+.0%} |")
        lines += ["", f"- Format choices (fused): {java['formats']['fused']}", "",
                  "Paper table: 8.7 / 8.8 / 6.7 s and 43.4 / 7.4 / 7.4 MB."]
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
