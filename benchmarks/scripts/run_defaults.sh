#!/bin/bash
# Table codec defaults on the Python server: snappy Parquet + uncompressed Arrow (today) against zstd + zstd.
# BIRD dev, fixed formats then the selector, on localhost and three emulated links.
# Needs results/bird/matrix_python_data_driven_static.csv (run_matrix.sh) to pick the largest results.
cd "$(dirname "$0")/../.." && source benchmarks/scripts/common.sh

BIRD=$PWD/data/dev_20240627
EXCL=518,701
OUT=results/defaults
mkdir -p $OUT
LARGE=$($PY - <<'PYEOF'
import csv
rows = csv.DictReader(open("results/bird/matrix_python_data_driven_static.csv"))
print(",".join(r["question_id"] for r in rows if int(r["n_rows"]) * int(r["n_cols"]) > 50000))
PYEOF
)

run_link() {
  local cfg=$1 pq=$2 ipc=$3 link=$4 delay=$5 mbps=$6 loss=$7
  stop_servers
  local net=()
  [ "$link" != localhost ] && net=(FORMAT_LATENCY_NETWORK_MBPS=$mbps)
  start_python_server defaults_${cfg}_${link} BIRD_SQLITE_ROOT=$BIRD PARQUET_ENCODING_STRATEGY=data_driven \
    PARQUET_COMPRESSION=$pq ARROW_IPC_COMPRESSION=$ipc "${net[@]}"
  local port=8000
  if [ "$link" != localhost ]; then
    nohup $PY -m benchmarks.netem.proxy --listen 8001 --delay-ms $delay --mbps $mbps --loss $loss > results/logs/proxy.log 2>&1 &
    sleep 2
    port=8001
  fi
  local sets=("stride:--stride 5" "large:--ids $LARGE")
  [ "$link" = localhost ] && sets=("all:")
  for set in "${sets[@]}"; do
    local tag=${set%%:*} sel=${set#*:}
    local out=$OUT/${cfg}_${link}_${tag}
    $PY -m benchmarks.sweeps.bird --exclude $EXCL $sel --port $port --out ${out}_static.csv
    $PY -m benchmarks.sweeps.selector --exclude $EXCL $sel --port $port --out ${out}_selector.csv
  done
  echo "$cfg $link done"
}

for link in "localhost 0 0 0" "lan 1 1000 0" "wan 40 50 0" "cellular 80 10 0.001"; do
  set -- $link
  run_link snappy_none snappy none $1 $2 $3 $4
  run_link zstd_zstd zstd zstd $1 $2 $3 $4
done
stop_servers
echo DEFAULTS_DONE
