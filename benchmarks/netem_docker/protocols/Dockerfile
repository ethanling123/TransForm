# Two images for the protocol experiment. Build from the repo root:
#   docker build -f benchmarks/netem_docker/protocols/Dockerfile --target server -t transform-proto-server .
#   docker build -f benchmarks/netem_docker/protocols/Dockerfile --target client -t transform-proto-client .
FROM golang:1.26-alpine AS build
WORKDIR /src
COPY benchmarks/netem_docker/protocols/go.mod benchmarks/netem_docker/protocols/main.go ./
RUN go mod tidy && CGO_ENABLED=0 go build -o /protoclient .

FROM caddy:2-alpine AS server
RUN apk add --no-cache iproute2-tc
COPY benchmarks/netem_docker/entrypoint.sh /entrypoint.sh
COPY benchmarks/netem_docker/protocols/Caddyfile /etc/caddy/Caddyfile
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["caddy", "run", "--config", "/etc/caddy/Caddyfile"]

FROM alpine:3.22 AS client
RUN apk add --no-cache iproute2-tc
COPY benchmarks/netem_docker/entrypoint.sh /entrypoint.sh
COPY --from=build /protoclient /protoclient
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
