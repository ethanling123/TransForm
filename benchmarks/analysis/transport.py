"""Tables for section 19 from results/netem_docker/transport_*.csv (transport.py) and protocols_*_*.csv (protocols/main.go).

    .venv/bin/python -m benchmarks.analysis.transport      # writes results/netem_docker/transport.md and protocols.md
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

R = Path("results/netem_docker")
PROFILES = [("unshaped", "unshaped"), ("lan", "LAN"), ("wan", "WAN"), ("wan_loss1", "WAN 1% loss"), ("cellular", "cellular"),
            ("badwifi", "bad wifi")]
CUT_ORDER = ["first", "10%", "50%", "100%"]
ROWS_PER_UNIT = 8192


def load(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open())) if path.exists() else []


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(out + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def med(rows, field):
    return statistics.median(float(r[field]) for r in rows) if rows else None


def iqr_pct(rows, field):
    """Interquartile range as a percentage of the median (a plain spread for the table notes)."""
    values = sorted(float(r[field]) for r in rows)
    if len(values) < 4:
        return None
    q = statistics.quantiles(values, n=4)
    return (q[2] - q[0]) / statistics.median(values) * 100 if statistics.median(values) else 0.0


def sel(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def fmt(x, digits=2):
    return "-" if x is None else f"{x:.{digits}f}"


def mb(x):
    return "-" if x is None else f"{x / 1e6:.2f}"


def transport_md() -> str:
    data = {p: load(R / f"transport_{p}.csv") for p, _ in PROFILES}
    data = {p: rows for p, rows in data.items() if rows}
    if not data:
        return "not run\n"
    labels = dict(PROFILES)
    tables_seen = sorted({r["table"] for rows in data.values() for r in rows})
    out = ["# Time to first rows, early termination and Arrow Flight", ""]
    for name in tables_seen:
        any_rows = next(r for rows in data.values() for r in rows if r["table"] == name)
        total_units = int(any_rows["total_units"])
        n_rows = int(next(r for rows in data.values() for r in rows if r["table"] == name and r["cut"] == "100%")["rows"])
        out += [f"#### Table `{name}` ({total_units} units of {ROWS_PER_UNIT} rows)", ""]

        out += ["Time until the first rows are decoded (median s, data plane only; blobs and JSON deliver nothing until the "
                "whole file is down and decoded).", ""]
        arms = ["json", "parquet_blob", "arrow_blob", "parquet_stream", "arrow_stream", "flight_none", "flight_lz4",
                "flight_zstd", "flight_zstd_info"]
        rows_out = []
        for arm in arms:
            cut = "first" if arm.endswith(("stream", "none", "lz4", "zstd", "info")) else "100%"
            cells = []
            for p in data:
                m = sel(data[p], table=name, arm=arm, cut=cut)
                cells.append(fmt(med(m, "ttfr_s"), 3))
            if any(c != "-" for c in cells):
                rows_out.append([arm] + cells)
        out += [table(["arm"] + [labels[p] for p in data], rows_out), ""]

        out += ["Whole table (median s to decode every row; MB that reached the client's NIC).", ""]
        rows_out = []
        for arm in arms:
            cells = []
            for p in data:
                m = sel(data[p], table=name, arm=arm, cut="100%")
                cells.append("-" if not m else f"{fmt(med(m, 't_cut_s'), 2)} s, {mb(med(m, 'rx_settled'))} MB")
            if any(c != "-" for c in cells):
                rows_out.append([arm] + cells)
        out += [table(["arm"] + [labels[p] for p in data], rows_out), ""]

        out += ["Stopping early: time to the target share of rows (median s), MB at the NIC when the client had them, "
                "MB once the link went quiet, and chunks or batches the server produced against the share needed.", ""]
        for p in data:
            rows_out = []
            for arm in ("parquet_stream", "arrow_stream", "flight_zstd", "flight_none"):
                for cut in CUT_ORDER:
                    m = sel(data[p], table=name, arm=arm, cut=cut)
                    if not m:
                        continue
                    target = 1 if cut == "first" else math.ceil({"10%": 0.1, "50%": 0.5, "100%": 1.0}[cut] * n_rows)
                    needed = min(total_units, math.ceil(target / ROWS_PER_UNIT))
                    rows_out.append([arm, cut, fmt(med(m, "t_cut_s"), 2), mb(med(m, "rx_cut")), mb(med(m, "rx_settled")),
                                     f"{med(m, 'server_units'):.0f} / {needed}"])
            out += [f"**{labels[p]}**", "", table(["arm", "stop at", "s", "MB at cut", "MB once quiet", "server units / needed"], rows_out), ""]
    return "\n".join(out)


def protocols_md() -> str:
    cells = {}
    for p, _ in PROFILES:
        for tag in ("single", "parallel8"):
            rows = load(R / f"protocols_{p}_{tag}.csv")
            if rows:
                cells[(p, tag)] = rows
    if not cells:
        return "not run\n"
    labels = dict(PROFILES)
    protos = ["plain", "h1", "h2", "h3"]
    names = {"plain": "HTTP/1.1", "h1": "HTTP/1.1 + TLS", "h2": "HTTP/2 + TLS", "h3": "HTTP/3"}
    out = ["# HTTP/1.1, HTTP/2 and HTTP/3", ""]

    def block(title, tag, mode, parallel, files):
        out.append(title)
        out.append("")
        header = ["profile", "payload"] + [names[x] for x in protos]
        rows_out = []
        for p, _ in PROFILES:
            rows = cells.get((p, tag))
            if not rows:
                continue
            for f in files:
                m = [r for r in rows if r["file"] == f and r["mode"] == mode and r["parallel"] == str(parallel)]
                if not m:
                    continue
                row = [labels[p], f]
                for proto in protos:
                    s = sel(m, proto=proto)
                    spread = iqr_pct(s, "wall_s")
                    row.append(fmt(med(s, "wall_s"), 3) + (f" (IQR {spread:.0f}%)" if spread is not None and spread > 20 else ""))
                rows_out.append(row)
        out.extend([table(header, rows_out), ""])

    files = ["p100k.parquet", "p1m.parquet", "p8m.parquet", "j1m.json"]
    out += ["Cells show the median of five repetitions; where the interquartile range exceeds 20% of the median it is shown.", ""]
    block("One request on a new connection: seconds until the payload is complete (median).", "single", "cold", 1, files)
    block("One request on a connection that has just carried the same payload (warm): seconds (median).", "single", "warm", 1, files)
    block("Eight simultaneous requests for the same payload on new connections: seconds until all eight are complete (median).",
          "parallel8", "cold", 8, ["p100k.parquet", "p1m.parquet"])

    out += ["Seconds until the first byte of the response (median, one request, new connection).", ""]
    rows_out = []
    for p, _ in PROFILES:
        rows = cells.get((p, "single"))
        if not rows:
            continue
        for f in ("p100k.parquet",):
            m = [r for r in rows if r["file"] == f and r["mode"] == "cold"]
            rows_out.append([labels[p], f] + [fmt(med(sel(m, proto=proto), "ttfb_s"), 3) for proto in protos])
    out += [table(["profile", "payload"] + [names[x] for x in protos], rows_out), ""]

    out += ["Bytes at the client's NIC (received plus sent, median) for one cold fetch of `p1m.parquet`, in kB.", ""]
    rows_out = []
    for p, _ in PROFILES:
        rows = cells.get((p, "single"))
        if not rows:
            continue
        m = [r for r in rows if r["file"] == "p1m.parquet" and r["mode"] == "cold"]
        row = [labels[p]]
        for proto in protos:
            s = sel(m, proto=proto)
            row.append("-" if not s else f"{(med(s, 'rx') + med(s, 'tx')) / 1e3:,.0f}")
        rows_out.append(row)
    out += [table(["profile"] + [names[x] for x in protos], rows_out), ""]
    return "\n".join(out)


def main():
    (R / "transport.md").write_text(transport_md())
    (R / "protocols.md").write_text(protocols_md())
    print("wrote transport.md and protocols.md")


if __name__ == "__main__":
    main()
