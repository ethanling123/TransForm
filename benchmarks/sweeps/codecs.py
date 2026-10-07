"""Table codec sweep: bytes, encode time and decode time for every Parquet and Arrow IPC arm.

In-process (no server, no network) so caching cannot blur the timings. Run it with the server environment, which
has pandas and the server's own `codec_selector`:

    python-server/.venv/bin/python -m benchmarks.sweeps.codecs

Datasets: the BIRD dev results larger than 50K cells (real data, many string columns) and TPC-DS catalog_sales slices
(numeric). Network times are modelled afterwards by `benchmarks.analysis.codecs` from the measured bytes and decode time.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sqlite3
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python-server"))
from codec_selector import select_encoding_params  # noqa: E402

PARQUET_CODECS = [
    ("none", None), ("snappy", None), ("lz4", None), ("gzip", None), ("brotli", None),
    ("zstd-1", 1), ("zstd-3", 3), ("zstd-9", 9), ("zstd-19", 19),
]
ARROW_CODECS = ["none", "lz4", "zstd"]
TPCDS_SHAPES = [(100_000, 34), (500_000, 34)]


def bird_tables(dev_json: str, big_csv: str, root: str):
    queries = {str(q["question_id"]): q for q in json.loads(Path(dev_json).read_text())}
    with open(big_csv) as f:
        ids = [r["question_id"] for r in csv.DictReader(f) if int(r["n_rows"]) * int(r["n_cols"]) > 50_000]
    for qid in ids:
        q = queries[qid]
        con = sqlite3.connect(f"{root}/dev_databases/{q['db_id']}/{q['db_id']}.sqlite")
        df = pd.read_sql_query(q["SQL"].replace("`", '"'), con)
        con.close()
        yield f"bird-{qid}", pa.Table.from_pandas(df, preserve_index=False)


def tpcds_tables(path: str):
    base = pq.read_table(path)
    for n_rows, n_cols in TPCDS_SHAPES:
        yield f"tpcds-{n_rows // 1000}Kx{n_cols}", base.slice(0, n_rows).select(base.column_names[:n_cols])


def parquet_arm(table, codec, level, strategy):
    kwargs = {"compression": codec.split("-")[0], "compression_level": level}
    if strategy == "data_driven":
        params = select_encoding_params(table)
        kwargs["use_dictionary"] = params.get("use_dictionary", [])
        if params.get("column_encoding"):
            kwargs["column_encoding"] = params["column_encoding"]
    return kwargs


def measure(encode, decode, reps):
    enc, dec, data = [], [], b""
    for _ in range(reps + 1):  # first run is warmup
        t = time.perf_counter()
        data = encode()
        enc.append(time.perf_counter() - t)
        t = time.perf_counter()
        decode(data)
        dec.append(time.perf_counter() - t)
    return len(data), statistics.median(enc[1:]), statistics.median(dec[1:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--tpcds", default="data/tpcds_catalog_sales_sf1.parquet")
    parser.add_argument("--bird-root", default="data/dev_20240627")
    parser.add_argument("--bird-ids-from", default="results/bird/matrix_python_data_driven_static.csv")
    parser.add_argument("--out", default="results/codecs/codecs.csv")
    args = parser.parse_args()

    tables = list(bird_tables(f"{args.bird_root}/dev.json", args.bird_ids_from, args.bird_root))
    tables += list(tpcds_tables(args.tpcds))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, table in tables:
        raw = table.nbytes
        for strategy in ("default", "data_driven"):
            for codec, level in PARQUET_CODECS:
                kwargs = parquet_arm(table, codec, level, strategy)

                def encode(kwargs=kwargs):
                    buf = io.BytesIO()
                    pq.write_table(table, buf, **kwargs)
                    return buf.getvalue()

                size, enc_s, dec_s = measure(encode, lambda b: pq.read_table(io.BytesIO(b)), args.reps)
                rows.append([name, table.num_rows, table.num_columns, raw, f"parquet/{strategy}", codec, size, enc_s, dec_s])
        for codec in ARROW_CODECS:
            opts = pa.ipc.IpcWriteOptions(compression=None if codec == "none" else codec)

            def encode(opts=opts):
                sink = pa.BufferOutputStream()
                with ipc.new_file(sink, table.schema, options=opts) as writer:
                    writer.write_table(table)
                return sink.getvalue().to_pybytes()

            size, enc_s, dec_s = measure(encode, lambda b: ipc.open_file(pa.BufferReader(b)).read_all(), args.reps)
            rows.append([name, table.num_rows, table.num_columns, raw, "arrow_ipc", codec, size, enc_s, dec_s])
        print(f"{name}: done ({table.num_rows} rows x {table.num_columns} cols, {raw / 1e6:.1f} MB in memory)", flush=True)

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "rows", "cols", "memory_bytes", "family", "codec", "bytes", "encode_s", "decode_s"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
