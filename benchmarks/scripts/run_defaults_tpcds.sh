#!/bin/bash
# TPC-DS catalog_sales through the Python server with snappy/none against zstd/zstd table defaults (localhost; the
# analysis adds modelled transfer time for slower links, since uploading the data through an emulated link is slow).
cd "$(dirname "$0")/../.." && source benchmarks/scripts/common.sh
mkdir -p results/defaults
for cfg in "snappy_none snappy none" "zstd_zstd zstd zstd"; do
  set -- $cfg
  stop_servers
  start_python_server defaults_tpcds_$1 PARQUET_ENCODING_STRATEGY=data_driven RESULT_CACHE_MAX_BYTES=4294967296 \
    PARQUET_COMPRESSION=$2 ARROW_IPC_COMPRESSION=$3
  $PY -m benchmarks.sweeps.tpcds --reps 3 --out results/defaults/tpcds_$1.csv > results/logs/defaults_tpcds_$1.log 2>&1
  echo "$1 done"
done
stop_servers
echo DEFAULTS_TPCDS_DONE
