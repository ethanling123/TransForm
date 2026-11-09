#!/bin/bash
# Early termination, time to first rows and an Arrow Flight baseline across a kernel-shaped link (see
# benchmarks/netem_docker/transport.py). Needs Docker running and data/tpcds_catalog_sales_sf1.parquet.
# Profiles are the paper's: name, one-way delay (ms), rate (Mbit/s), loss (%).
#   benchmarks/scripts/run_transport.sh [profile ...]        default: unshaped lan wan cellular badwifi
#   TRANSPORT_ARGS="--reps 3 --tables tpcds" benchmarks/scripts/run_transport.sh wan
cd "$(dirname "$0")/../.." || exit 1
mkdir -p results/netem_docker
COMPOSE="docker compose -p transform-transport -f benchmarks/netem_docker/compose_transport.yml"

docker build -q -f benchmarks/netem_docker/Dockerfile -t transform-netem . || exit 1
shape() {  # delay_ms rate_mbit loss_pct
  case $1 in
    unshaped) echo "" ;; lan) echo "1 1000 0" ;; wan) echo "40 50 0" ;; cellular) echo "80 10 0.1" ;; badwifi) echo "30 5 1" ;;
    *) echo "unknown profile $1" >&2; exit 1 ;;
  esac
}
for profile in ${@:-unshaped lan wan cellular badwifi}; do
  set -- $(shape $profile)
  json=""; case $profile in unshaped|lan) json="--json" ;; esac
  PROFILE=$profile NETEM_DELAY_MS=${1:-} NETEM_RATE_MBIT=${2:-} NETEM_LOSS_PCT=${3:-} \
    TRANSPORT_ARGS="$json $TRANSPORT_ARGS" \
    $COMPOSE up --abort-on-container-exit --exit-code-from client 2>&1 | grep -E "^client-1 +\||netem on"
  $COMPOSE down -v >/dev/null 2>&1
done
