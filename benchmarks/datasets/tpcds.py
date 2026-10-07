"""Generate TPC-DS catalog_sales (scale factor 1) as Parquet using DuckDB's dsdgen."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sf", type=float, default=1)
    parser.add_argument("--out", default="data/tpcds_catalog_sales_sf1.parquet")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL tpcds; LOAD tpcds;")
    con.execute(f"CALL dsdgen(sf={args.sf})")
    con.execute(f"COPY (SELECT * FROM catalog_sales) TO '{args.out}' (FORMAT parquet)")
    n_rows, n_cols = con.execute("SELECT count(*) FROM catalog_sales").fetchone()[0], len(
        con.execute("DESCRIBE catalog_sales").fetchall()
    )
    print(f"catalog_sales sf={args.sf}: {n_rows} rows x {n_cols} cols -> {args.out}")


if __name__ == "__main__":
    main()
