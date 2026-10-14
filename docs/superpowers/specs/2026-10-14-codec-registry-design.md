# F1: arm registry and pluggable codecs (Python server)

## Goal
Replace the hard-coded arm list with a registry so F2 (image/audio arms) and F4 (bandit) can add arms without touching
the selector or the routes. Scope is the Python server only. Selector behaviour does not change, so BIRD bytes and
latency stay identical. Out of scope: media codecs (F2), selector or bandit changes (F4), signature and digest
enforcement (F5), the Java server.

## Today
Arm names are repeated by hand in `format_selector.FormatName`, `format_mab.FORMATS`, the `large_*` tools, the hint
tables in `describe_result_formats`, the `large_result_auto` dispatch and one HTTP route per arm. Parquet and Arrow IPC
compression are process-wide settings (`PARQUET_COMPRESSION`, `ARROW_IPC_COMPRESSION`), not per-arm.

## Design
**`python-server/arms.py`** (no dependency on `server_app.py`)
- `ArmSpec(name, modality, container, transport, codec, level, lossless, selectable)`, frozen, with `mime_type` and
  `describe()`. `codec=None` means the server's process-wide default for the container (the existing `parquet_blob`
  and `arrow_ipc_blob` arms).
- Encoders are plain functions: `encode_parquet(table, codec, level, strategy)` and `encode_arrow_ipc(table, codec)`
  (moved out of `server_app`, which keeps thin wrappers), `encode_raw(data, codec, level)`. A `Codec` class
  hierarchy was not built; there is one encoder per container and nothing yet needs polymorphism (F2 can add one
  when a second implementation of the same container appears).
- `ArmRegistry`: `register`, `get`, `names(selectable_only=False)`, `for_modality`. One module-level `REGISTRY`.
- Arms: the 10 existing names (all `selectable=True`) plus new table arms, `selectable=False`:
  `parquet_{none,gzip,lz4,zstd1,zstd3,zstd9}_blob`, `arrow_ipc_{lz4,zstd}_blob`.

**Selection stays closed.** `FormatName` and `FORMATS` are unchanged; a test asserts
`REGISTRY.names(selectable_only=True) == set(FORMATS) == set(FormatName args)`. Hint tables list selectable arms only.

**Data plane.** `ResultConfig.arm` records the arm a handle was minted for. `GET /data/{handle}` resolves it,
runs the arm's encoder and sets `Content-Type`, `X-Benchmark-Bytes`, `X-Payload-Codec`, `X-Uncompressed-Bytes` and
`X-Content-Digest` (sha256). Inline arms (`json`, `text_inline`) return 400 there. Stream arms reuse the existing chunk
generators. The legacy routes stay as aliases to the same handlers with the same bytes.

**Descriptors.** Existing fields (`mode`, `id`, `url`, `format`, ...) are unchanged. Added: `modality`, `mime_type`,
`codec`, `data_url`, and `size` and `digest` when the encoded bytes are already cached (so tool calls do no extra
encoding).

**Tools.** `list_arms()` returns the registry. `fetch_arm(result_id, arm)` returns a descriptor for any registered
blob arm, including non-selectable ones.

## Done when
1. For every existing blob arm, the legacy route, `/data/{handle}` and a direct encoder call return identical bytes
   (BIRD, synthetic and unstructured fixtures).
2. The paper harness fused arm on the Python server reproduces `results/paper_repro/run1` per-query wire bytes
   (4.83 MB, `data_driven`) with latency within noise.
3. Every new arm round-trips through its decoder (tests). Its size matches the table codec sweep: checked by hand on
   `tpcds-100Kx34` (all eight arms equal to the byte); the encoder is also tested against a plain `pq.write_table`.
4. The selector never returns a non-selectable arm; the drift test passes.
5. Existing tests and the Java parity check still pass.

## Result (2026-10-14)
- Golden digests for all existing arms are unchanged; 37 tests pass.
- Paper harness, fused arm, `data_driven`: wire bytes and chosen format identical for all 1534 queries. Transport
  time 3.41-3.43 s (two runs) against 3.30 s for the pre-registry code rerun under the same machine load (+4%); the
  JSON path, which the change does not touch, moved by the same amount.
- Tool parity with the Java server: the 18 existing tools match; `list_arms` and `fetch_arm` exist only in Python.
- The separate spec-review step was skipped at the user's request ("write the spec and build it").
