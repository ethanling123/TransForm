# TransForm Python server

Python implementation of the TransForm MCP server: a control plane (small JSON over MCP) plus a data plane
(Parquet, Arrow IPC, raw and gzip payloads over HTTP), with a format selector that picks the representation
per result. It exposes the same 18 MCP tools, with the same argument schemas, as `../java-server/`
(checked with `python -m benchmarks.analysis.parity --a <python-port> --b <java-port>` from the repo root).

Restored from the original Python implementation (commit `55de98c`) and changed in three places:
- duplicate SQL column names are renamed (`name`, `name_1`) instead of failing,
- mixed-type SQLite columns are cast to strings, and invalid UTF-8 text is replaced,
- the `describe_and_fetch_result` tool was added (Java had it, Python did not).

It runs on the stock `mcp` package; the vendored SDK fork from the old tree is not needed.

## Run

```bash
cd python-server
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
BIRD_SQLITE_ROOT=/path/to/dev_20240627 PARQUET_ENCODING_STRATEGY=data_driven \
  .venv/bin/uvicorn server_app:app --host 127.0.0.1 --port 8000
```

MCP endpoint: `http://localhost:8000/mcp/mcp`. The client and sweep scripts are in `../client/`.

## Configuration (environment)

| Variable | Default | Meaning |
|---|---|---|
| `BIRD_SQLITE_ROOT` | none | folder holding `dev_databases/<db>/<db>.sqlite` (BIRD and Spider layout) |
| `PARQUET_COMPRESSION` | `zstd` | Parquet codec (`snappy` was the default before the zstd comparison in REPORT.md) |
| `PARQUET_ENCODING_STRATEGY` | `default` | `data_driven` picks per-column encodings (the paper's setting) |
| `ARROW_IPC_COMPRESSION` | `zstd` | `none`, `lz4` or `zstd`; a reader needs zstd support in its Arrow build (pyarrow has it) |
| `FORMAT_COST_AWARE` | off | `1` enables the experimental cost-aware unstructured selector (`min_latency` only, needs `FORMAT_LATENCY_NETWORK_MBPS` > 0): sizes are sampled estimates, nothing is compressed to choose, and compute time is scored with transfer time (see `cost_model.py`) |
| `FORMAT_INLINE_MAX_BYTES` | `0` (off) | experimental: a blob arm whose bytes are at most this many is returned inside the tool result's `_meta` (key `transform/inline`, base64 with a sha256) instead of behind a URL. Measured gain only up to about 100 KB on WAN and 50 KB on cellular; 65536 is a reasonable value. Hosts that forward `_meta` to the model would put the base64 in context |
| `FORMAT_LEARN_LINK` | off | `1`: with no `FORMAT_LATENCY_NETWORK_MBPS`, use the median throughput of the last ten client-reported downloads of 200 KB or more (`record_format_outcome` with `bytes` and `latency_s`); one estimate for the whole server |
| `FORMAT_LATENCY_RTT_MS` | `0` | with a link speed set, inline only while the extra base64 transfer time is less than this round trip |
| `FORMAT_SELECT_TARGET` | `min_latency` | `min_bytes`, `min_latency` or `min_time_to_first_rows` |
| `FORMAT_LATENCY_NETWORK_MBPS` | `0` | link rate the latency target assumes; at 0 it falls back to bytes |
| `SQLITE_QUERY_TIMEOUT_S` | `30` | queries slower than this are interrupted |
| `TRANSFORM_SCRATCH_DIR` | the source folder | where generated files go (`data/materialized/`, `results/`); the benchmark scripts use `results/scratch/` |
| `MCP_API_KEY` | unset | when set, every request needs `Authorization: Bearer <key>` |

## Unstructured payload arms

Results registered with `POST /materialized-raw` (images, audio, text, any bytes) are served by five arms, and
`large_result_auto` picks one from the exact sizes (`describe_result_formats` shows them):

| Arm | Route | Notes |
|---|---|---|
| `text_inline` | none (in the tool result) | text up to 256 KB; for the min_bytes and min_latency targets, texts under 4 KB stay inline |
| `raw_blob` | `/raw/{id}` | as uploaded |
| `gzip_blob` | `/raw-gzip/{id}` | sent with `Content-Encoding: gzip`, so HTTP clients decode it themselves |
| `zstd_blob` | `/raw-zstd/{id}` | zstd level 3 |
| `brotli_blob` | `/raw-brotli/{id}` | brotli level 5 |

The zstd and brotli arms carry no `Content-Encoding` header. The response sets `X-Payload-Codec`,
`X-Uncompressed-Bytes` and `X-Original-Content-Type`, and the `large_result_auto` descriptor repeats the codec and the
uncompressed size. A client decompresses with the codec and the size, for example
`pyarrow.Codec("zstd").decompress(body, decompressed_size=n)`.

Payloads over 1 MB are sampled first: if the first 256 KB barely compresses (PNG, JPEG, Opus, zip), the codec arms are
left out of the hints and `raw_blob` wins, instead of compressing something that will not shrink. These arms exist only
in the Python server; the Java server has `text_inline`, `raw_blob` and `gzip_blob` only.

## Arm registry

Every way of delivering a result is an arm, defined once in [arms.py](arms.py) (`ArmSpec`: modality, container, codec,
level, transport, `selectable`). The selector and the bandit only see selectable arms; the rest are fetched on request.

- `list_arms` returns the registry. `fetch_arm(arm, result_id)` returns a descriptor for any blob arm.
- Registered but never chosen by the selector: `parquet_{none,gzip,lz4,zstd1,zstd3,zstd9}_blob`,
  `arrow_ipc_{lz4,zstd}_blob`.
- `GET /data/{handle}` serves any minted handle by the arm it was minted for and adds `X-Payload-Codec` and
  `X-Content-Digest` (sha256). The old routes (`/blobs`, `/ipc-blobs`, `/streams`, `/raw*`) are unchanged.
- Descriptors keep their old fields and add `format`, `modality`, `mime_type`, `codec`, `data_url`, and `size` and
  `digest` when the encoded bytes are already in memory.
- `tests/golden_arm_bytes.json` holds digests of what every existing arm served before the registry; the tests fail if
  those bytes change.

## Known limitations

- A query that returns zero rows is rejected ("Empty result"); there is nothing to transport.
- Gold queries slower than the timeout fail (BIRD dev 518 and 701 take 32 s and 168 s in plain SQLite).
- The security gaps listed in `../REPORT.md` (shared bearer key, no result expiry, wide-open CORS) are unchanged.

## Model-aware arms (experimental, Python only)

`large_result_auto` takes an optional `consumer`: `"vision"` returns an image scaled to a vision model's input limits
(long edge 1568 px and 1568 visual tokens, the Claude standard tier; the original is kept when scaling would not make the
file smaller), `"asr"` returns 16 kHz mono 16-bit WAV. The arms are `image_fit_blob` and `audio_16k_blob`, registered
as lossy and never chosen by the selector. They need Pillow (in `requirements.txt`) and `ffmpeg` on the PATH. The extra
argument exists only in this server, so the tool schema no longer matches the Java server's for `large_result_auto`. The
provider's own resampler differs from Pillow's, so "the model sees the same thing" is approximate. Measured on the 212
M3-Bench images and 14 speech clips: `benchmarks/sweeps/model_aware.py`.

## Conditional and ranged fetches

The raw, Parquet and Arrow IPC blob routes send an `ETag` (the handle and the byte count, so no hashing per response), answer `If-None-Match` with `304`, and
accept a single `Range: bytes=a-b` header (`206`). Measured in REPORT.md section 17: a repeat of an unchanged 4.5 MB
result takes 0.1 s instead of 0.9 s on WAN; parallel ranges did not make a download faster.

