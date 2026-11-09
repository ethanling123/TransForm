"""Does a delivery arm's output size reveal a secret to someone who can write rows into the same result? (compression side channel)

Each trial builds a table with one secret value and three attacker rows holding a guess, encodes it with the arm, and compares
the output size for the right guess against a wrong guess of the same shape. Attacker accuracy is the share of trials where the
right guess gives the smaller output (ties count half), so 50% is no leak and 100% is a perfect oracle. The table shows max(accuracy, 1 - accuracy): an attacker can calibrate the
direction on data they control, and some encodings invert it (the right guess encodes larger). Secret kinds: a digit
string, a 9-digit integer, and a rare enum value. Layouts: `same` (secret and guess in one column), `cross` (secret in one
column, guess in another) and `far` (same column, but the guess is 200k rows after the secret, past the first data page).
The encoders are the server's own (`arms.encode_parquet`, `arms.encode_arrow_ipc`); the JSON arms are the table as JSON,
plain and gzipped as one body.

    .venv/bin/python -m benchmarks.analysis.leakage --trials 150 --far-trials 30 --out results/security/leakage.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python-server"))
import arms  # noqa: E402

STATUSES = [f"status_{i}" for i in range(20)]
RARE = [f"dx_{i:03d}" for i in range(50)]
FAR_ROWS = 200_000
KINDS = ("string", "integer", "enum")
ARROW_TYPE = {"string": pa.string(), "integer": pa.int64(), "enum": pa.string()}


def encodings() -> dict:
    out = {}
    for codec in ("zstd", "snappy", "gzip", "none"):
        for strategy in ("default", "data_driven"):
            out[f"parquet {codec} {strategy}"] = lambda t, c=codec, s=strategy: arms.encode_parquet(t, c, None, s)
    for codec in ("zstd", "lz4", "none"):
        out[f"arrow_ipc {codec}"] = lambda t, c=codec: arms.encode_arrow_ipc(t, c)
    out["json inline"] = lambda t: json.dumps(t.to_pylist()).encode()
    out["json gzip body"] = lambda t: gzip.compress(json.dumps(t.to_pylist()).encode(), mtime=0)
    return out


FAR_ENCODINGS = ("parquet zstd default", "parquet zstd data_driven", "parquet none default", "parquet none data_driven",
                 "arrow_ipc zstd", "json gzip body")


def filler(kind: str, n: int, rnd: random.Random) -> list:
    if kind == "string":
        return [f"order {i} shipped via {rnd.choice(('ground', 'air', 'sea'))}" for i in range(n)]
    if kind == "integer":
        return [rnd.randrange(10**9) for _ in range(n)]
    return [rnd.choice(STATUSES) for _ in range(n)]


def secret_value(kind: str, rnd: random.Random):
    if kind == "string":
        return f"ssn={rnd.randrange(10**9):09d}"
    if kind == "integer":
        return rnd.randrange(10**9)
    return rnd.choice(RARE)


def build(kind: str, layout: str, base: list, secret, guess, pos: int) -> pa.Table:
    tail = [guess] * 3
    if layout == "cross":
        secret_col = [None] * (len(base) + 3)
        secret_col[pos] = secret
        return pa.table({"secret": pa.array(secret_col, type=ARROW_TYPE[kind]), "other": pa.array(base + tail, type=ARROW_TYPE[kind])})
    column = list(base)
    column[pos] = secret
    return pa.table({"v": pa.array(column + tail, type=ARROW_TYPE[kind])})


def accuracy(encode, kind: str, layout: str, trials: int, seed: int) -> float:
    rnd = random.Random(seed)
    score = 0.0
    for _ in range(trials):
        n = FAR_ROWS if layout == "far" else rnd.randint(150, 400)
        base, pos = filler(kind, n, rnd), rnd.randrange(50, 150)
        secret = secret_value(kind, rnd)
        wrong = secret_value(kind, rnd)
        while wrong == secret:
            wrong = secret_value(kind, rnd)
        shape = "same" if layout == "far" else layout
        right_size = len(encode(build(kind, shape, base, secret, secret, pos)))
        wrong_size = len(encode(build(kind, shape, base, secret, wrong, pos)))
        score += 1.0 if right_size < wrong_size else 0.5 if right_size == wrong_size else 0.0
    return score / trials


def table_md(rows: dict, layouts: tuple) -> str:
    cols = [(k, l) for l in layouts for k in KINDS]
    head = "| arm | " + " | ".join(f"{k} / {l}" for k, l in cols) + " |"
    lines = [head, "|---|" + "---|" * len(cols)]
    for name, cells in rows.items():
        lines.append(f"| {name} | " + " | ".join(f"{max(cells[f'{k}/{l}'], 1 - cells[f'{k}/{l}']):.0%}" if f"{k}/{l}" in cells else "" for k, l in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--far-trials", type=int, default=30)
    ap.add_argument("--out", default="results/security/leakage.json")
    args = ap.parse_args()

    results: dict = {}
    for name, encode in encodings().items():
        results[name] = {}
        layouts = [("same", args.trials), ("cross", args.trials)] + ([("far", args.far_trials)] if name in FAR_ENCODINGS else [])
        for layout, trials in layouts:
            for kind in KINDS:
                results[name][f"{kind}/{layout}"] = accuracy(encode, kind, layout, trials, seed=7)
        print(name, {k: f"{v:.0%}" for k, v in results[name].items()}, flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"trials": args.trials, "far_trials": args.far_trials, "far_rows": FAR_ROWS, "accuracy": results}, indent=1))
    markdown = table_md(results, ("same", "cross", "far"))
    out.with_suffix(".md").write_text(markdown + "\n")
    print("\n" + markdown)


if __name__ == "__main__":
    main()
