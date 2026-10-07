"""Summarize the table codec sweep: size ranking and the best arm under each emulated network profile.

Modelled time = transfer (bytes / bandwidth) + decode, optionally + encode. The one-way delay is the same for
every arm, so it does not change the ranking and is left out.
"""

from __future__ import annotations

import csv
from pathlib import Path

PROFILES = [("LAN", 1000), ("WAN", 50), ("Cellular", 10), ("BadWifi", 5)]  # Mbit/s, from the workshop paper


def load(path="results/codecs/codecs.csv"):
    rows = []
    for r in csv.DictReader(open(path)):
        rows.append({**r, "bytes": int(r["bytes"]), "encode_s": float(r["encode_s"]), "decode_s": float(r["decode_s"]),
                     "memory_bytes": int(r["memory_bytes"]), "arm": f"{r['family']} {r['codec']}"})
    return rows


def modelled(r, mbps, with_encode):
    return r["bytes"] * 8 / (mbps * 1e6) + r["decode_s"] + (r["encode_s"] if with_encode else 0.0)


def main():
    rows = load()
    datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    print("#### Smallest and largest output per dataset (MB; in-memory Arrow size for reference)\n")
    print("| dataset | rows x cols | in memory | smallest arm | MB | largest arm | MB |\n|---|---|---|---|---|---|---|")
    for d in datasets:
        rs = [r for r in rows if r["dataset"] == d]
        lo, hi = min(rs, key=lambda r: r["bytes"]), max(rs, key=lambda r: r["bytes"])
        print(f"| {d} | {rs[0]['rows']} x {rs[0]['cols']} | {rs[0]['memory_bytes'] / 1e6:.1f} | {lo['arm']} | "
              f"{lo['bytes'] / 1e6:.2f} | {hi['arm']} | {hi['bytes'] / 1e6:.2f} |")

    for with_encode, title in ((False, "pre-encoded (transfer + decode)"), (True, "encoded per request (encode + transfer + decode)")):
        print(f"\n#### Best arm per network profile, modelled time, {title}\n")
        print("| dataset | " + " | ".join(p for p, _ in PROFILES) + " |\n|---|" + "---|" * len(PROFILES))
        for d in datasets:
            rs = [r for r in rows if r["dataset"] == d]
            cells = []
            for _, mbps in PROFILES:
                best = min(rs, key=lambda r: modelled(r, mbps, with_encode))
                cells.append(f"{best['arm']} ({modelled(best, mbps, with_encode) * 1e3:.0f} ms)")
            print(f"| {d} | " + " | ".join(cells) + " |")

    print("\n#### One dataset in full: tpcds-500Kx34 (bytes MB, encode ms, decode ms)\n")
    print("| arm | MB | encode ms | decode ms |\n|---|---|---|---|")
    for r in sorted((r for r in rows if r["dataset"] == "tpcds-500Kx34"), key=lambda r: r["bytes"]):
        print(f"| {r['arm']} | {r['bytes'] / 1e6:.2f} | {r['encode_s'] * 1e3:.0f} | {r['decode_s'] * 1e3:.0f} |")


if __name__ == "__main__":
    main()
