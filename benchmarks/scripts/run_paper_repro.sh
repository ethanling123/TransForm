#!/usr/bin/env bash
# Run the workshop paper's own harness (benchmarks/paper_harness, unmodified from commit 55de98c) against the
# Python server, once per arm, on the full BIRD dev set with gold SQL. Settings follow
# results/RUNBOOK_benchmarks_full_bird_network.md in that commit.
#
#   [SERVER=java] [ARMS="both server"] [STRATEGY=default|data_driven] benchmarks/scripts/run_paper_repro.sh [tag]   (tag names the output dir, default run1)
cd "$(dirname "$0")/../.." && source benchmarks/scripts/common.sh
TAG=${1:-run1}
OUT=results/paper_repro/$TAG
mkdir -p "$OUT"
export PYTHONPATH=$PWD:$PWD/python-server:$PWD/benchmarks/paper_harness FORMAT_SELECT_TARGET=min_latency

for arm in ${ARMS:-both server inline}; do
  stop_servers
  if [ "${SERVER:-python}" = java ]; then
    start_java_server paper_repro_$arm SQLITE_QUERY_TIMEOUT_S=30 BIRD_SQLITE_ROOT=$PWD/data/dev_20240627
  else
    start_python_server paper_repro_$arm PARQUET_ENCODING_STRATEGY=${STRATEGY:-data_driven} SQLITE_QUERY_TIMEOUT_S=30 \
      BIRD_SQLITE_ROOT=$PWD/data/dev_20240627
  fi
  $PY benchmarks/paper_harness/bench_bird_e2e.py --data-dir data/dev_20240627 --bird-questions dev.json \
    --max-queries 2000 --arms $arm --results $OUT/$arm.jsonl --overwrite > results/logs/paper_repro_${TAG}_$arm.client.log 2>&1
  $PY benchmarks/paper_harness/summarize_bird_e2e.py --input $OUT/$arm.jsonl --json-out $OUT/${arm}_summary.json \
    --md $OUT/${arm}_summary.md > /dev/null 2>&1
done
clean_scratch
