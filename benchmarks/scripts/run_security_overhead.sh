#!/usr/bin/env bash
# Quick security-overhead runs: the BIRD sweep (every 10th query) and the unshaped network-check suite, with and without
# TRANSFORM_SECURE=1, twice each in alternating order so drift shows up. Outputs go to results/security_overhead/.
cd "$(dirname "$0")/../.." && source benchmarks/scripts/common.sh
OUT=results/security_overhead
mkdir -p $OUT results/scratch results/logs
export PYTHONPATH=$PWD
BIRD=$PWD/data/dev_20240627

for round in 1 2; do
  for mode in off on; do
    stop_servers
    if [ $mode = on ]; then export TRANSFORM_SECURE=1; start_python_server so_$mode BIRD_SQLITE_ROOT=$BIRD TRANSFORM_SECURE=1
    else unset TRANSFORM_SECURE; start_python_server so_$mode BIRD_SQLITE_ROOT=$BIRD; fi
    $PY -m benchmarks.sweeps.bird --stride 10 --out $OUT/bird_${mode}_$round.csv > $OUT/bird_${mode}_$round.log 2>&1
    $PY -m benchmarks.netem_docker.quick --reps 3 --out $OUT/quick_${mode}_$round.csv > $OUT/quick_${mode}_$round.log 2>&1
    clean_scratch
  done
done
stop_servers
unset TRANSFORM_SECURE
