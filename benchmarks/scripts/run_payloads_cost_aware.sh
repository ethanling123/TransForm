#!/bin/bash
# The unstructured payload sweep with the experimental cost-aware selector (FORMAT_COST_AWARE=1) on the Python server.
# Same payloads and forced arms as run_payloads.sh, so each CSV carries its own fresh baseline; the selector min_latency
# rows are the cost-aware ones. Localhost is told it has a 20 Gbit/s link (the replay's assumption): with no link speed
# the selector cannot weigh transfer against compute.
cd "$(dirname "$0")/../.." && source benchmarks/scripts/common.sh

BIRD=$PWD/data/dev_20240627
OUT=results/payloads_cost_aware
mkdir -p $OUT

stop_servers
start_python_server payloads_ca_localhost FORMAT_COST_AWARE=1 FORMAT_LATENCY_NETWORK_MBPS=20000 BIRD_SQLITE_ROOT=$BIRD PARQUET_ENCODING_STRATEGY=data_driven
$PY -m benchmarks.sweeps.payloads --out $OUT/localhost.csv > results/logs/payloads_ca_localhost.log 2>&1
echo "localhost done"

for p in "lan 1 1000 0" "wan 40 50 0" "cellular 80 10 0.001"; do
  set -- $p
  case " ${PROFILES:-lan wan cellular} " in *" $1 "*) ;; *) continue ;; esac
  stop_servers
  start_python_server payloads_ca_$1 FORMAT_COST_AWARE=1 FORMAT_LATENCY_NETWORK_MBPS=$3 BIRD_SQLITE_ROOT=$BIRD PARQUET_ENCODING_STRATEGY=data_driven
  nohup $PY -m benchmarks.netem.proxy --listen 8001 --delay-ms $2 --mbps $3 --loss $4 > results/logs/proxy.log 2>&1 &
  sleep 2
  $PY -m benchmarks.sweeps.payloads --port 8001 --image-stride 9 --out $OUT/$1.csv > results/logs/payloads_ca_$1.log 2>&1
  echo "$1 done"
done
clean_scratch
echo PAYLOADS_CA_DONE
