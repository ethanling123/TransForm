"""How often do BIRD dev gold queries return a result another query already returned? An upper bound on what a
conditional fetch (If-None-Match) or content-addressed reuse could save on a workload shaped like this one.

Results are hashed as sorted multisets of rows, so column order and row order do not matter. Queries 518 and 701 (over the
30 s timeout) are skipped. This is the gold-SQL set, not an agent's session: an agent would repeat itself more, or less.

    .venv/bin/python -m benchmarks.analysis.reuse_census
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

ROOT = Path("data/dev_20240627")
SKIP = {"518", "701"}


def main():
    queries = json.loads((ROOT / "dev.json").read_text())
    seen = {}
    per_db = defaultdict(lambda: [0, 0])
    total = dup = total_bytes = dup_bytes = failed = 0
    for q in queries:
        qid = str(q["question_id"])
        if qid in SKIP:
            continue
        con = sqlite3.connect(f"file:{ROOT}/dev_databases/{q['db_id']}/{q['db_id']}.sqlite?mode=ro", uri=True)
        con.text_factory = lambda b: b.decode("utf-8", "replace")
        con.execute("PRAGMA query_only = ON")
        try:
            rows = con.execute(q["SQL"].replace("`", '"')).fetchall()
        except Exception:
            failed += 1
            continue
        finally:
            con.close()
        canon = sorted(json.dumps([str(v) for v in r]) for r in rows)
        body = "\n".join(canon).encode()
        key = (q["db_id"], hashlib.sha256(body).hexdigest())
        total += 1
        total_bytes += len(body)
        per_db[q["db_id"]][0] += 1
        if key in seen:
            dup += 1
            dup_bytes += len(body)
            per_db[q["db_id"]][1] += 1
        seen[key] = qid
    lines = ["# Result reuse across BIRD dev gold queries", "",
             f"{total} queries ran ({failed} failed). {dup} ({dup / total:.1%}) returned a result an earlier query in the same "
             f"database already returned; those results are {dup_bytes / 1e6:.2f} MB of {total_bytes / 1e6:.2f} MB "
             f"({dup_bytes / total_bytes:.1%}) of the canonical row bytes.", ""]
    lines += ["| database | queries | repeated results |", "|---|---|---|"]
    lines += [f"| {db} | {n} | {d} |" for db, (n, d) in sorted(per_db.items(), key=lambda kv: -kv[1][1])]
    text = "\n".join(lines) + "\n"
    Path("results/data_plane").mkdir(parents=True, exist_ok=True)
    Path("results/data_plane/reuse_census.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
