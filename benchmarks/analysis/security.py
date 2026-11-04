"""Summarise the security work: the attack suite after each defence, and the injection experiment on a model.

Reads results/attacks/{before_open,before_keyed,step1..step5}.json and results/security/injection.json, writes
results/security/summary.md (included in REPORT.md by benchmarks/analysis/report.py).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from benchmarks.analysis.injection import INJECTIONS, wilson

R = Path("results")
STEPS = [("step1", "1 principals"), ("step2", "2 signed links"), ("step3", "3 signed descriptors"), ("step4", "4 origin allowlist"),
         ("step5", "5 envelope")]
SHORT = {"VULNERABLE": "works", "NOT_REPRODUCED": "blocked", "ERROR": "error"}


def load(path):
    p = R / path
    return {r["id"]: r for r in json.loads(p.read_text())} if p.exists() else {}


def attack_table() -> str:
    columns = [("before_open", "open"), ("before_keyed", "keyed")] + STEPS
    data = {name: load(f"attacks/{name}.json") for name, _ in columns}
    titles = next((d for d in data.values() if d), {})
    out = ["| ID | Threat | " + " | ".join(label for _, label in columns) + " |", "|---|---|" + "---|" * len(columns)]
    for tid in sorted(titles):
        cells = [SHORT.get(data[name].get(tid, {}).get("outcome", ""), "-") for name, _ in columns]
        out.append(f"| {tid} | {titles[tid]['threat']} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def injection_tables() -> str:
    path = R / "security" / "injection.json"
    if not path.exists():
        return "not run"
    d = json.loads(path.read_text())
    rows = d["rows"]
    out = [f"Model: `{d['model']}`, {d['samples']} samples per cell, temperature 0.7. Attack success is the canary token in the reply; "
           "utility is the correct answer in the reply.", "",
           "| Envelope | Data | Attack success (95% CI) | Correct on clean data | Correct under attack |", "|---|---|---|---|---|"]
    for condition in ("none", "boundary", "datamark"):
        for kind in ("text", "table"):
            cell = [r for r in rows if r["condition"] == condition and r["kind"] == kind]
            attacked = [r for r in cell if r["injection"] != "clean"]
            clean = [r for r in cell if r["injection"] == "clean"]
            k, n = sum(r["attacked"] for r in attacked), len(attacked)
            lo, hi = wilson(k, n)
            out.append(f"| {condition} | {kind} | {k}/{n} = {k / n:.0%} ({lo:.0%}-{hi:.0%}) | "
                       f"{sum(r['correct'] for r in clean)}/{len(clean)} | {sum(r['correct'] for r in attacked)}/{n} |")
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["injection"] != "clean":
            by[(r["condition"], r["injection"])][0] += r["attacked"]
            by[(r["condition"], r["injection"])][1] += 1
    out += ["", "Attack success by injection, text and table pooled:", "",
            "| Injection | none | boundary | datamark |", "|---|---|---|---|"]
    for name in INJECTIONS:
        out.append(f"| {name} | " + " | ".join(f"{by[(c, name)][0]}/{by[(c, name)][1]}" for c in ("none", "boundary", "datamark")) + " |")
    return "\n".join(out)


def main():
    text = f"""# Security defences

{attack_table()}

{injection_tables()}
"""
    (R / "security").mkdir(parents=True, exist_ok=True)
    (R / "security" / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
