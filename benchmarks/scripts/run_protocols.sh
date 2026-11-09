#!/bin/bash
# HTTP/1.1, HTTP/1.1 + TLS, HTTP/2 and HTTP/3 across a kernel-shaped link (see benchmarks/netem_docker/protocols/).
# Needs Docker running and data/tpcds_catalog_sales_sf1.parquet. Profiles: name, one-way delay (ms), rate (Mbit/s), loss (%).
#   benchmarks/scripts/run_protocols.sh [profile ...]    default: lan wan wan_loss1 cellular badwifi
#   REPS=3 benchmarks/scripts/run_protocols.sh wan
cd "$(dirname "$0")/../.." || exit 1
mkdir -p results/netem_docker results/scratch
COMPOSE="docker compose -p transform-protocols -f benchmarks/netem_docker/compose_protocols.yml"
REPS=${REPS:-5}

.venv/bin/python -m benchmarks.netem_docker.make_payloads data/tpcds_catalog_sales_sf1.parquet results/scratch/protocol_payloads || exit 1
for target in server client; do
  docker build -q -f benchmarks/netem_docker/protocols/Dockerfile --target $target -t transform-proto-$target . || exit 1
done
shape() {  # delay_ms rate_mbit loss_pct
  case $1 in
    lan) echo "1 1000 0" ;; wan) echo "40 50 0" ;; wan_loss1) echo "40 50 1" ;; cellular) echo "80 10 0.1" ;; badwifi) echo "30 5 1" ;;
    *) echo "unknown profile $1" >&2; exit 1 ;;
  esac
}
for profile in ${@:-lan wan wan_loss1 cellular badwifi}; do
  read d r l <<< "$(shape $profile)"
  # one request: all payloads, cold and warm; eight at once: the two smaller Parquet payloads, cold
  PROFILE=$profile NETEM_DELAY_MS=$d NETEM_RATE_MBIT=$r NETEM_LOSS_PCT=$l TAG=single \
    PROTO_ARGS="-reps $REPS -files p100k.parquet,p1m.parquet,p8m.parquet,j1m.json -parallels 1 -modes cold,warm" \
    $COMPOSE up --abort-on-container-exit --exit-code-from client 2>&1 | grep -E "netem on|rep [0-9]+:"
  $COMPOSE down -v >/dev/null 2>&1
  PROFILE=$profile NETEM_DELAY_MS=$d NETEM_RATE_MBIT=$r NETEM_LOSS_PCT=$l TAG=parallel8 \
    PROTO_ARGS="-reps $REPS -files p100k.parquet,p1m.parquet -parallels 8 -modes cold" \
    $COMPOSE up --abort-on-container-exit --exit-code-from client 2>&1 | grep -E "netem on|rep [0-9]+:"
  $COMPOSE down -v >/dev/null 2>&1
done
