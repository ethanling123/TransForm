#!/usr/bin/env bash
# Run the T1-T9 attack demos against the Python server with TRANSFORM_SECURE=1 and two principals (alice, bob).
#   benchmarks/scripts/run_attacks_defended.sh [tag]      # tag names the output file (default: defended)
cd "$(dirname "$0")/../.." && source benchmarks/scripts/common.sh
TAG=${1:-defended}
mkdir -p results/attacks
export PYTHONPATH=$PWD

stop_servers
start_python_server attacks_defended TRANSFORM_SECURE=1 TRANSFORM_URL_TTL_S=3 MCP_API_KEYS=alice:sk-alice-test-key,bob:sk-bob-test-key
$PY -m benchmarks.attacks.run_attacks --label defended --api-key sk-alice-test-key --second-key sk-bob-test-key \
  --out results/attacks/${TAG}.json
clean_scratch
