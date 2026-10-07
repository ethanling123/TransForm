#!/bin/bash
# Reproduce every measurement in REPORT.md, then rebuild it. Takes a few hours.
# Set SKIP_JAVA=1 to skip the Java server runs (the report's Java columns then stay as they are).
cd "$(dirname "$0")/../.." && source benchmarks/scripts/common.sh

# BIRD: Python (and Java) x Parquet encoding
if [ -n "$SKIP_JAVA" ]; then IMPLS=python benchmarks/scripts/run_matrix.sh; else benchmarks/scripts/run_matrix.sh; fi

# Spider dev
benchmarks/scripts/run_dataset.sh spider data/spider_data/dev.json data/spider_root query

# TPC-DS catalog_sales and the synthetic sweep. The Python server is run with the default 64 MB payload cache
# and with a 4 GB cache.
tpcds_python() {  # <cache-bytes> <out.csv>
  stop_servers
  start_python_server tpcds PARQUET_ENCODING_STRATEGY=data_driven RESULT_CACHE_MAX_BYTES=$1
  $PY -m benchmarks.sweeps.tpcds --reps 3 --out $2 > results/logs/tpcds.log 2>&1
}
tpcds_python 67108864 results/tpcds/python.csv
tpcds_python 4294967296 results/tpcds/python_bigcache.csv
stop_servers
start_python_server synthetic PARQUET_ENCODING_STRATEGY=data_driven
$PY -m benchmarks.sweeps.synthetic --reps 3 --out results/synthetic/python.csv > results/logs/synthetic.log 2>&1

if [ -z "$SKIP_JAVA" ]; then
  stop_servers
  start_java_server tpcds_java PARQUET_ENCODING_STRATEGY=data_driven
  $PY -m benchmarks.sweeps.tpcds --reps 3 --out results/tpcds/java.csv > results/logs/tpcds_java.log 2>&1
  $PY -m benchmarks.sweeps.synthetic --reps 3 --out results/synthetic/java.csv > results/logs/synthetic_java.log 2>&1
fi

benchmarks/scripts/run_network.sh > results/logs/network.log 2>&1

# table codec sweep (in-process, needs the server environment for pandas)
python-server/.venv/bin/python -m benchmarks.sweeps.codecs > results/logs/codecs.log 2>&1

# images, speech and text: in-process media codecs, then the unstructured arms through the server
[ -d data/audio ] || $PY -m benchmarks.datasets.speech   # macOS only (uses `say`); needs ffmpeg
benchmarks/scripts/run_media.sh
benchmarks/scripts/run_payloads.sh > results/logs/payloads.log 2>&1

# tool parity between the two servers (ports 8000 = Python, 8002 = Java)
if [ -z "$SKIP_JAVA" ]; then
  stop_servers
  start_python_server parity_python BIRD_SQLITE_ROOT=$PWD/data/dev_20240627
  (cd results/scratch && env BIRD_SQLITE_ROOT=$PWD/../../data/dev_20240627 nohup java --add-opens=java.base/java.nio=ALL-UNNAMED \
    -jar ../../$JAR --server.port=8002 > ../logs/parity_java.log 2>&1 &)
  sleep 12
  $PY -m benchmarks.analysis.parity --a 8000 --b 8002 | tee results/parity.txt
  stop_servers
fi

clean_scratch
$PY -m benchmarks.analysis.report
echo ALL_DONE
