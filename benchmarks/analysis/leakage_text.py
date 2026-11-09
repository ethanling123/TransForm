"""Compression side channel on the unstructured arms (text and raw files), the counterpart of `leakage.py` for tables.

A document holds one secret line and, elsewhere, three attacker lines holding a guess. The arm is the server's own
(`arms.encode_raw` with the codec and level of `gzip_blob`, `zstd_blob` and `brotli_blob`; `raw_blob` and `text_inline` are
uncompressed). Layouts: `near` (about 20 KB of prose between the secret and the guess), `far` (about 2 MB between them, past
gzip's 32 KB window), and `chunked` (the secret and the guess are compressed as two separate blobs and the sizes added, the
unstructured analogue of separate columns). Attacker accuracy is the share of trials where the right guess gives the smaller
output (ties count half); the table shows max(accuracy, 1 - accuracy).

    .venv/bin/python -m benchmarks.analysis.leakage_text --trials 100 --out results/security/leakage_text.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python-server"))
import arms  # noqa: E402

WORDS = ("quarter revenue customer invoice shipment delay refund meeting budget forecast vendor contract policy audit "
         "release incident rota cluster ticket review roadmap hiring onboarding payment ledger archive notes summary").split()
CODECS = {
    "gzip_blob": lambda b: arms.encode_raw(b, "gzip", None),
    "zstd_blob (level 3)": lambda b: arms.encode_raw(b, "zstd", 3),
    "brotli_blob (level 5)": lambda b: arms.encode_raw(b, "brotli", 5),
    "raw_blob / text_inline": lambda b: b,
}
GAPS = {"near": 20_000, "far": 2_000_000}


def prose(nbytes: int, rnd: random.Random) -> str:
    out, size = [], 0
    while size < nbytes:
        line = " ".join(rnd.choice(WORDS) for _ in range(rnd.randint(8, 16))) + "."
        out.append(line)
        size += len(line) + 1
    return "\n".join(out)


def digits(rnd: random.Random) -> str:
    return f"{rnd.randrange(10**9):09d}"


def trial_sizes(encode, layout: str, rnd: random.Random):
    secret, wrong = digits(rnd), digits(rnd)
    head, gap = prose(5_000, rnd), prose(GAPS.get(layout, 20_000), rnd)

    def size(guess: str) -> int:
        victim = f"{head}\nssn={secret}\n"
        attacker = f"{gap}\nssn={guess}\nssn={guess}\nssn={guess}\n"
        if layout == "chunked":
            return len(encode(victim.encode())) + len(encode(attacker.encode()))
        return len(encode((victim + attacker).encode()))

    return size(secret), size(wrong)


def accuracy(encode, layout: str, trials: int) -> float:
    rnd = random.Random(11)
    score = 0.0
    for _ in range(trials):
        right, wrong = trial_sizes(encode, layout, rnd)
        score += 1.0 if right < wrong else 0.5 if right == wrong else 0.0
    return score / trials


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--out", default="results/security/leakage_text.json")
    args = ap.parse_args()

    results = {name: {layout: accuracy(encode, layout, args.trials) for layout in ("near", "far", "chunked")}
               for name, encode in CODECS.items()}
    lines = ["| arm | near (20 KB gap) | far (2 MB gap) | chunked |", "|---|---|---|---|"]
    for name, cells in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{max(cells[l], 1 - cells[l]):.0%}" for l in ("near", "far", "chunked")) + " |")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"trials": args.trials, "accuracy": results}, indent=1))
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
