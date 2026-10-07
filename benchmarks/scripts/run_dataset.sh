#!/bin/bash
# Fixed formats and the selector for one text-to-SQL dataset on the Python server (data_driven Parquet).
# usage: run_dataset.sh <name> <dev.json> <sqlite-root> <sql-field> [exclude-ids]
#   e.g. run_dataset.sh spider data/spider_data/dev.json data/spider_root query
# Writes results/<name>/python_{static,selector}.csv.
cd "$(dirname "$0")/../.." && source benchmarks/scripts/common.sh

NAME=$1; DEV=$2; ROOT=$(cd "$3" && pwd); FIELD=$4; EXCL=${5:-}
mkdir -p results/$NAME

stop_servers
start_python_server dataset_$NAME BIRD_SQLITE_ROOT=$ROOT PARQUET_ENCODING_STRATEGY=data_driven
log=results/logs/dataset_$NAME.log
$PY -m benchmarks.sweeps.bird --dev $DEV --sql-field $FIELD --exclude "$EXCL" --out results/$NAME/python_static.csv > $log 2>&1
$PY -m benchmarks.sweeps.selector --dev $DEV --sql-field $FIELD --exclude "$EXCL" --out results/$NAME/python_selector.csv >> $log 2>&1
stop_servers
echo "$NAME done"
