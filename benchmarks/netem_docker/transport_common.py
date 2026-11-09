"""Tables and constants shared by the Flight server, the probe server and the transport client."""

from __future__ import annotations

import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROWS = int(os.environ.get("TRANSPORT_ROWS", "200000"))
ROWS_PER_CHUNK = 8192
FLIGHT_PORT = 8815
TPCDS_PATH = os.environ.get("TPCDS_PATH", "/data/tpcds.parquet")


def mixed_table(n: int) -> pa.Table:
    rng = np.random.default_rng(0)
    words = np.array([f"category-{i}" for i in range(40)])
    names = ["".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), 12)) for _ in range(n)]
    return pa.table({
        "id": np.arange(n),
        "price": rng.random(n) * 1000,
        "quantity": rng.integers(0, 500, n),
        "category": words[rng.integers(0, 40, n)],
        "name": names,
        "created": (1_700_000_000 + rng.integers(0, 90_000_000, n)).astype("datetime64[s]"),
    })


def tpcds_table(n: int) -> pa.Table:
    batch = next(pq.ParquetFile(TPCDS_PATH).iter_batches(batch_size=n))
    return pa.Table.from_batches([batch])


def tables() -> dict[str, pa.Table]:
    return {"tpcds": tpcds_table(ROWS).combine_chunks(), "mixed": mixed_table(ROWS).combine_chunks()}
