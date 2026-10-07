# Shared helpers for the run scripts. Source it from the repo root:
#   cd "$(dirname "$0")/../.." && source benchmarks/scripts/common.sh
PY=.venv/bin/python
JAR=java-server/target/multimodal-mcp-0.1.0-SNAPSHOT.jar
mkdir -p results/logs results/scratch results/bird results/spider results/tpcds results/synthetic results/network

stop_servers() {
  pkill -f "uvicorn server_app"
  pkill -f multimodal-mcp-0.1.0
  pkill -f benchmarks.netem.proxy
  sleep 2
}

# start_python_server <log-name> [VAR=value ...]   (port 8000, snappy Parquet, uncompressed Arrow unless overridden)
# The long keep-alive timeout stops the server closing a connection while a slow emulated link is still delivering a response.
start_python_server() {
  local log=$1; shift
  (cd results/scratch && env PARQUET_COMPRESSION=snappy ARROW_IPC_COMPRESSION=none TRANSFORM_SCRATCH_DIR=$PWD "$@" \
    nohup ../../python-server/.venv/bin/uvicorn server_app:app --app-dir ../../python-server \
    --host 127.0.0.1 --port 8000 --timeout-keep-alive 600 > ../logs/$log.log 2>&1 &)
  sleep 12
}

# start_java_server <log-name> [VAR=value ...]     (port 8000)
start_java_server() {
  local log=$1; shift
  (cd results/scratch && env PARQUET_COMPRESSION=snappy ARROW_IPC_COMPRESSION=none "$@" \
    nohup java --add-opens=java.base/java.nio=ALL-UNNAMED -jar ../../$JAR --server.port=8000 > ../logs/$log.log 2>&1 &)
  sleep 12
}

# Servers keep every payload they are given (and compressed copies); a payload run leaves gigabytes behind.
clean_scratch() {
  stop_servers
  rm -rf results/scratch/data results/scratch/results
}
