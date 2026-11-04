"""What do the defences cost on the existing benchmarks? Compares TRANSFORM_SECURE off and on from
benchmarks/scripts/run_security_overhead.sh (BIRD sweep every 10th query, and the unshaped network-check suite), two rounds each.

Writes results/security_overhead/summary.md (included in REPORT.md section 18).
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

R = Path("results/security_overhead")
ROUNDS = (1, 2)


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def ms(x):
    return f"{x * 1000:.1f}"


def bird_table() -> str:
    cols = [("mint", "parquet_mint_s"), ("fetch", "parquet_fetch_s"), ("mint", "arrow_ipc_mint_s"), ("fetch", "arrow_ipc_fetch_s")]
    out = ["| Arm | Part | off, round 1 | off, round 2 | on, round 1 | on, round 2 | on / off |", "|---|---|---|---|---|---|---|"]
    data = {(m, r): rows(R / f"bird_{m}_{r}.csv") for m in ("off", "on") for r in ROUNDS}
    for arm, (part, col) in zip(("parquet_blob",) * 2 + ("arrow_ipc_blob",) * 2, cols):
        med = {k: statistics.median(float(x[col]) for x in v) for k, v in data.items()}
        off, on = statistics.mean([med[("off", r)] for r in ROUNDS]), statistics.mean([med[("on", r)] for r in ROUNDS])
        out.append(f"| {arm} | {part} (median ms) | " + " | ".join(ms(med[(m, r)]) for m in ("off", "on") for r in ROUNDS)
                   + f" | {on / off:.2f}x |")
    same = all(a["parquet_bytes"] == b["parquet_bytes"] and a["arrow_ipc_bytes"] == b["arrow_ipc_bytes"]
               for a, b in zip(data[("off", 1)], data[("on", 1)]))
    n = len(data[("off", 1)])
    return "\n".join(out) + f"\n\n{n} BIRD queries (every 10th), median per query; payload bytes identical with the defences on: {same}."


def quick_table() -> str:
    data = {(m, r): rows(R / f"quick_{m}_{r}.csv") for m in ("off", "on") for r in ROUNDS}
    out = ["| Workload | Strategy | Bytes off / on | off (ms, mean of rounds) | on (ms) | on / off |", "|---|---|---|---|---|---|"]
    keys = [(x["workload"], x["strategy"]) for x in data[("off", 1)] if x["workload"] != "link"]
    for key in keys:
        def pick(mode):
            vals, size = [], None
            for r in ROUNDS:
                for x in data[(mode, r)]:
                    if (x["workload"], x["strategy"]) == key:
                        vals.append(float(x["seconds"]))
                        size = int(x["wire_bytes"])
            return statistics.mean(vals), size
        (off, off_bytes), (on, on_bytes) = pick("off"), pick("on")
        out.append(f"| {key[0]} | {key[1]} | {off_bytes:,} / {on_bytes:,} | {ms(off)} | {ms(on)} | {on / off:.2f}x |")
    return "\n".join(out)


def client_costs() -> str:
    """The checks the benchmark clients above do not run: the client's sha256 over the payload and one Ed25519 verification."""
    import hashlib
    import os
    import time

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    data = os.urandom(8_000_000)
    best = min((lambda t: (hashlib.sha256(data).digest(), time.perf_counter() - t)[1])(time.perf_counter()) for _ in range(5))
    key = Ed25519PrivateKey.generate()
    message = os.urandom(300)
    sig = key.sign(message)
    public = key.public_key()
    t = time.perf_counter()
    for _ in range(2000):
        public.verify(sig, message)
    verify_s = (time.perf_counter() - t) / 2000
    t = time.perf_counter()
    for _ in range(2000):
        key.sign(message)
    sign_s = (time.perf_counter() - t) / 2000
    return (f"Checks the benchmark clients above do not run, measured separately: client sha256 over a payload runs at "
            f"{len(data) / best / 1e6:,.0f} MB/s ({best / len(data) * 1e9:.2f} ms per MB); verifying one grant signature takes "
            f"{verify_s * 1e6:.0f} us and signing one takes {sign_s * 1e6:.0f} us.")


def main():
    text = f"""# Security overhead (localhost, unshaped)

{bird_table()}

{quick_table()}

{client_costs()}
"""
    (R / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
