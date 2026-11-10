# One image for both machines: the Python server (its own venv, mcp 2.x) and the client (mcp 1.x).
# Build from the repo root:  docker build -f benchmarks/netem_docker/Dockerfile -t transform-netem .
FROM python:3.14-slim
RUN apt-get update && apt-get install -y --no-install-recommends iproute2 iputils-ping socat \
    && rm -rf /var/lib/apt/lists/*
COPY python-server/requirements.txt /tmp/server-requirements.txt
RUN python -m venv /opt/server && /opt/server/bin/pip install --no-cache-dir -r /tmp/server-requirements.txt
RUN python -m venv /opt/client && /opt/client/bin/pip install --no-cache-dir mcp==1.30.0 httpx pyarrow numpy
WORKDIR /app
COPY python-server /app/python-server
COPY client /app/client
COPY benchmarks /app/benchmarks
COPY benchmarks/netem_docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
