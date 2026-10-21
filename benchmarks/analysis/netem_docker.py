"""Summarize the two-container, kernel-shaped network check (results/netem_docker/*.csv) as markdown.

    .venv/bin/python -m benchmarks.analysis.netem_docker
"""

from __future__ import annotations

import csv
from pathlib import Path

D = Path("results/netem_docker")
PROFILES = [("unshaped", "unshaped"), ("lan", "LAN 1 ms, 1 Gbit"), ("wan", "WAN 40 ms, 50 Mbit"),
            ("cellular", "cellular 80 ms, 10 Mbit, 0.1% loss"), ("badwifi", "bad wifi 30 ms, 5 Mbit, 1% loss")]
CAPS = {"unshaped": None, "lan": 1000, "wan": 50, "cellular": 10, "badwifi": 5}


def load(name):
    path = D / f"{name}.csv"
    return list(csv.DictReader(open(path))) if path.exists() else None


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(out + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def main():
    data = {p: load(p) for p, _ in PROFILES}
    data = {p: rows for p, rows in data.items() if rows}
    labels = dict(PROFILES)
    link = []
    for p, rows in data.items():
        ping, download = [r for r in rows if r["workload"] == "link"]
        mbit = float(download["strategy"].split("=")[1].split()[0])
        cap = CAPS[p]
        link.append([labels[p], f"{float(ping['seconds']) * 1000:.0f} ms", f"{mbit:.0f} Mbit/s", f"{mbit / cap:.0%}" if cap else "-"])
    out = ["# Real network between two containers (quick check)", "", "#### Link check", "",
           "Round trip of an MCP ping (it includes opening a connection and the HTTP exchange, so it is more than the "
           "shaped round trip) and the rate of one 4 MB download against the configured cap.", "",
           table(["profile", "MCP ping", "4 MB download", "share of the cap"], link), ""]
    workloads = []
    for r in next(iter(data.values())):
        if r["workload"] != "link" and (r["workload"], r["strategy"]) not in workloads:
            workloads.append((r["workload"], r["strategy"]))
    out += ["#### Latency by workload and profile", "",
            "Seconds to fetch and decode (median of two runs after a warm-up), with the bytes that crossed the link. "
            "Rows named `selector` show the arm the selector chose.", ""]
    for workload in dict.fromkeys(w for w, _ in workloads):
        rows = []
        for _, strategy in [x for x in workloads if x[0] == workload]:
            cells = []
            for p, profile_rows in data.items():
                r = next(x for x in profile_rows if x["workload"] == workload and x["strategy"] == strategy)
                cells.append(f"{float(r['seconds']):.2f} s" + (f" ({r['arm']})" if r["arm"] else ""))
            wire = next(x for x in next(iter(data.values())) if x["workload"] == workload and x["strategy"] == strategy)["wire_bytes"]
            rows.append([strategy, f"{int(wire) / 1e3:,.0f} KB", *cells])
        out += [f"##### {workload}", "", table(["strategy", "wire", *[labels[p] for p in data]], rows), ""]
    text = "\n".join(out) + "\n"
    (D / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
