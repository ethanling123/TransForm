#!/bin/bash
# BIRD dev on {python, java} x {default, data_driven} Parquet encoding: fixed formats, then the selector.
# Set IMPLS=python (or java) to run one server only.
cd "$(dirname "$0")/../.." && source benchmarks/scripts/common.sh

BIRD=$PWD/data/dev_20240627
EXCL=518,701   # gold SQL slower than the 30 s query timeout (32 s and 168 s in plain SQLite)

for impl in ${IMPLS:-python java}; do
  for enc in data_driven default; do
    stop_servers
    start_${impl}_server matrix_${impl}_${enc} BIRD_SQLITE_ROOT=$BIRD PARQUET_ENCODING_STRATEGY=$enc
    log=results/logs/matrix_${impl}_${enc}.log
    $PY -m benchmarks.sweeps.bird --exclude $EXCL --out results/bird/matrix_${impl}_${enc}_static.csv > $log 2>&1
    $PY -m benchmarks.sweeps.selector --exclude $EXCL --out results/bird/matrix_${impl}_${enc}_selector.csv >> $log 2>&1
    echo "$impl $enc done"
  done
done
stop_servers
echo MATRIX_DONE
