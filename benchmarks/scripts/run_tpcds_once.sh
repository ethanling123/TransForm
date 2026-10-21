#!/bin/bash
# TPC-DS catalog_sales through the Python server with the current code and zstd table defaults (the same settings as the
# zstd_zstd run of run_defaults_tpcds.sh), written to the CSV named by $1. Extra VAR=value arguments go to the server.
cd "$(dirname "$0")/../.." && source benchmarks/scripts/common.sh
OUT=$1; shift
stop_servers
start_python_server tpcds_once PARQUET_ENCODING_STRATEGY=data_driven RESULT_CACHE_MAX_BYTES=4294967296 \
  PARQUET_COMPRESSION=zstd ARROW_IPC_COMPRESSION=zstd "$@"
$PY -m benchmarks.sweeps.tpcds --reps 3 --out $OUT > results/logs/tpcds_once.log 2>&1
stop_servers
echo "$OUT done"
