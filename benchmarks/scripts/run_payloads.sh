#!/bin/bash
# Unstructured payloads (images, speech, text) through the Python server: localhost, then the four network profiles.
# The network runs use every 9th image to keep transfer time reasonable on the slow links.
cd "$(dirname "$0")/../.." && source benchmarks/scripts/common.sh

BIRD=$PWD/data/dev_20240627
mkdir -p results/payloads

# PROFILES="badwifi" reruns only those profiles; SKIP_LOCALHOST=1 skips the unemulated run.
if [ -z "$SKIP_LOCALHOST" ]; then
  stop_servers
  start_python_server payloads BIRD_SQLITE_ROOT=$BIRD PARQUET_ENCODING_STRATEGY=data_driven
  $PY -m benchmarks.sweeps.payloads --out results/payloads/localhost.csv > results/logs/payloads_localhost.log 2>&1
  echo "localhost done"
fi

# profiles from the workshop paper: name, delay (ms), bandwidth (Mbit/s), packet loss
for p in "lan 1 1000 0" "wan 40 50 0" "cellular 80 10 0.001" "badwifi 30 5 0.01"; do
  set -- $p
  case " ${PROFILES:-lan wan cellular badwifi} " in *" $1 "*) ;; *) continue ;; esac
  stop_servers
  start_python_server payloads FORMAT_LATENCY_NETWORK_MBPS=$3 BIRD_SQLITE_ROOT=$BIRD PARQUET_ENCODING_STRATEGY=data_driven
  nohup $PY -m benchmarks.netem.proxy --listen 8001 --delay-ms $2 --mbps $3 --loss $4 > results/logs/proxy.log 2>&1 &
  sleep 2
  $PY -m benchmarks.sweeps.payloads --port 8001 --image-stride 9 --out results/payloads/$1.csv > results/logs/payloads_$1.log 2>&1
  echo "$1 done"
done
clean_scratch
echo PAYLOADS_DONE
