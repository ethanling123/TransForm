#!/usr/bin/env bash
# Run the T1-T9 attack demos against the Python server with no API key and with a shared key.
#   benchmarks/scripts/run_attacks.sh [tag]      # tag names the output files (default: before)
cd "$(dirname "$0")/../.." && source benchmarks/scripts/common.sh
TAG=${1:-before}
mkdir -p results/attacks
export PYTHONPATH=$PWD

stop_servers
start_python_server attacks_open
$PY -m benchmarks.attacks.run_attacks --label open --out results/attacks/${TAG}_open.json

stop_servers
start_python_server attacks_keyed MCP_API_KEY=sk-attack-test-key
$PY -m benchmarks.attacks.run_attacks --label keyed --api-key sk-attack-test-key --out results/attacks/${TAG}_keyed.json
clean_scratch
