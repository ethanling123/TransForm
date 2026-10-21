#!/bin/bash
# Quick format check across a real, kernel-shaped link between two containers (server and client); see
# benchmarks/netem_docker/. Needs Docker running. Profiles are the paper's: name, one-way delay (ms), rate (Mbit/s), loss (%).
#   benchmarks/scripts/run_netem_docker.sh [profile ...]        default: unshaped lan wan cellular
#   FORMAT_COST_AWARE=1 benchmarks/scripts/run_netem_docker.sh wan
cd "$(dirname "$0")/../.." || exit 1
mkdir -p results/netem_docker
COMPOSE="docker compose -f benchmarks/netem_docker/compose.yml"

docker build -q -f benchmarks/netem_docker/Dockerfile -t transform-netem . || exit 1
shape() {  # delay_ms rate_mbit loss_pct
  case $1 in
    unshaped) echo "" ;; lan) echo "1 1000 0" ;; wan) echo "40 50 0" ;; cellular) echo "80 10 0.1" ;; badwifi) echo "30 5 1" ;;
    *) echo "unknown profile $1" >&2; exit 1 ;;
  esac
}
for profile in ${@:-unshaped lan wan cellular}; do
  set -- $(shape $profile)
  PROFILE=$profile NETEM_DELAY_MS=${1:-} NETEM_RATE_MBIT=${2:-} NETEM_LOSS_PCT=${3:-} \
    $COMPOSE up --abort-on-container-exit --exit-code-from client 2>&1 | grep -E "^client-1 +\||netem on"
  $COMPOSE down -v >/dev/null 2>&1
done
