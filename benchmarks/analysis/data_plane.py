"""Tables for the data-plane experiments run in the two-container network (results/netem_docker/*.csv).

    .venv/bin/python -m benchmarks.analysis.data_plane
"""

from __future__ import annotations

import csv
from pathlib import Path

D = Path("results/netem_docker")


def load(name):
    path = D / f"{name}.csv"
    return list(csv.DictReader(open(path))) if path.exists() else None


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(out + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def inline():
    rows = []
    for prof, label in (("lan", "LAN"), ("wan", "WAN"), ("cellular", "cellular")):
        off, on = load(f"breakdown2_{prof}_off") or load(f"breakdown_{prof}_off"), load(f"breakdown2_{prof}_on") or load(f"breakdown_{prof}_on")
        if not (off and on):
            continue
        for a in (r for r in off if r["strategy"].startswith("selector call")):
            b = next(r for r in on if r["workload"] == a["workload"] and r["strategy"] == a["strategy"])
            t0, t1 = float(a["seconds"]), float(b["seconds"])
            rows.append([label, a["workload"].replace("breakdown ", ""), f"{int(a['wire_bytes']) / 1e3:,.0f} KB", f"{t0:.3f} s", f"{t1:.3f} s", f"{(t1 / t0 - 1):+.0%}",
                         "yes" if "inline" in b["arm"] else "no (over cap)"])
    return table(["link", "result", "wire", "descriptor + fetch", "inline in _meta", "change", "inlined"], rows)


def ranges():
    rows = []
    for prof, label in (("lan", "LAN"), ("wan", "WAN"), ("cellular", "cellular")):
        data = [r for r in load(f"ranges_{prof}") or [] if r["workload"] == "ranges 8 MB"]
        if data:
            rows.append([label, *[r["strategy"].split("=")[1].strip() for r in data]])
    return table(["link", "plain GET", "1 range", "2 ranges", "4 ranges", "8 ranges"], rows)


def learned():
    rows = []
    for prof, label in (("lan", "LAN (1000 Mbit/s)"), ("wan", "WAN (50 Mbit/s)"), ("cellular", "cellular (10 Mbit/s)")):
        parts = {}
        for tag in ("learned", "configured", "none"):
            data = load(f"learn_{prof}_{tag}")
            if data:
                parts[tag] = {(r["workload"], r["strategy"].split(" =")[0] if r["workload"] == "learned link" else r["strategy"]): r for r in data}
        if not parts:
            continue
        est = next(r["strategy"] for r in load(f"learn_{prof}_learned") if r["workload"] == "learned link").replace("server estimate = ", "")
        for workload in ("text 600 KB (compressible)", "bytes 1 MB (incompressible)"):
            cells = []
            for tag in ("learned", "configured", "none"):
                r = parts.get(tag, {}).get((workload, "selector min_latency"))
                cells.append(f"{r['arm']} {float(r['seconds']):.2f} s" if r else "-")
            rows.append([label, est if workload.startswith("text") else "", workload, *cells])
    return table(["link", "learned estimate", "payload", "link speed learned", "link speed configured", "no link speed"], rows)


def conditional():
    rows = []
    for prof, label in (("wan", "WAN"), ("cellular", "cellular")):
        data = [r for r in load(f"conditional_{prof}") or [] if r["workload"] == "conditional fetch"]
        if data:
            a, b = data
            rows.append([label, f"{float(a['seconds']):.2f} s", f"{float(b['seconds']):.2f} s", f"{1 - float(b['seconds']) / float(a['seconds']):.0%}"])
    return table(["link", "repeat, whole payload", "repeat, If-None-Match (304)", "time saved"], rows)


def main():
    text = "\n".join(["# Data-plane experiments in the two-container network", "",
                      "#### Inline bytes in the tool result against descriptor plus fetch (median of five, uploaded table as Parquet or Arrow)", "", inline(), "",
                      "#### One 8 MB payload over one connection or several byte ranges", "", ranges(), "",
                      "#### Selector choices when the server learns the link speed from client reports", "", learned(), "",
                      "#### Fetching an unchanged 4.5 MB result again (synthetic)", "", conditional(), ""])
    (D / "data_plane.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
