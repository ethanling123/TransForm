#!/bin/bash
# Network-emulated sweeps on the Python server: {BIRD, Spider} x {LAN, WAN, Cellular, BadWifi}.
# Needs results/bird/matrix_python_data_driven_static.csv (run_matrix.sh) to pick the largest BIRD results.
cd "$(dirname "$0")/../.." && source benchmarks/scripts/common.sh

BIRD=$PWD/data/dev_20240627
SPIDER=$PWD/data/spider_root
LARGE=$($PY - <<'PYEOF'
import csv
rows = csv.DictReader(open("results/bird/matrix_python_data_driven_static.csv"))
print(",".join(r["question_id"] for r in rows if int(r["n_rows"]) * int(r["n_cols"]) > 50000))
PYEOF
)

# run_profile <profile> <one-way-delay-ms> <mbps> <loss> <dataset> <dev.json> <sqlite-root> <sql-field> <exclude-ids>
run_profile() {
  local prof=$1 delay=$2 mbps=$3 loss=$4 ds=$5 dev=$6 root=$7 field=$8 excl=$9
  stop_servers
  start_python_server network FORMAT_LATENCY_NETWORK_MBPS=$mbps BIRD_SQLITE_ROOT=$root PARQUET_ENCODING_STRATEGY=data_driven
  nohup $PY -m benchmarks.netem.proxy --listen 8001 --delay-ms $delay --mbps $mbps --loss $loss > results/logs/proxy.log 2>&1 &
  sleep 2
  local sets=("stride:--stride 5")
  [ "$ds" = bird ] && sets+=("large:--ids $LARGE")
  for set in "${sets[@]}"; do
    local tag=${set%%:*} sel=${set#*:}
    local out=results/network/${prof}_${ds}_${tag}
    $PY -m benchmarks.sweeps.bird --dev $dev --sql-field $field --exclude "$excl" $sel --port 8001 --out ${out}_static.csv
    $PY -m benchmarks.sweeps.selector --dev $dev --sql-field $field --exclude "$excl" $sel --port 8001 --out ${out}_selector.csv
  done
  echo "$prof $ds done"
}

# profiles from the workshop paper: name, delay (ms), bandwidth (Mbit/s), packet loss
for p in "lan 1 1000 0" "wan 40 50 0" "cellular 80 10 0.001" "badwifi 30 5 0.01"; do
  set -- $p
  run_profile $1 $2 $3 $4 bird $BIRD/dev.json $BIRD SQL 518,701
  run_profile $1 $2 $3 $4 spider $PWD/data/spider_data/dev.json $SPIDER query ""
done
stop_servers
echo NETWORK_DONE
