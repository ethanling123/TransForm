"""Turn the attack-demo JSON files into one table.

    .venv/bin/python -m benchmarks.analysis.attacks --tag before
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="before")
    parser.add_argument("--dir", default="results/attacks")
    args = parser.parse_args()

    runs = {label: {r["id"]: r for r in json.loads(Path(args.dir, f"{args.tag}_{label}.json").read_text())}
            for label in ("open", "keyed")}
    lines = [f"# Attack demos T1-T9 ({args.tag})", "",
             "`open` = no API key configured; `keyed` = shared bearer key (the only auth the server has). "
             "VULNERABLE = the attack worked; NOT_REPRODUCED = it failed here.", "",
             "| ID | Threat | Handoff expected | open | keyed |", "|---|---|---|---|---|"]
    for tid, r in runs["open"].items():
        lines.append(f"| {tid} | {r['threat']} | {r['expected_before']} | {r['outcome']} | {runs['keyed'][tid]['outcome']} |")
    lines += ["", "## Evidence", ""]
    for label, rows in runs.items():
        lines.append(f"### {label}")
        lines += [f"- **{tid}** {r['outcome']}: {r['evidence']}" for tid, r in rows.items()]
        lines.append("")
    Path(args.dir, f"{args.tag.upper()}.md").write_text("\n".join(lines))
    print("\n".join(lines[:16]))


if __name__ == "__main__":
    main()
