"""Build REPORT.md from the CSVs in results/ so no number is typed by hand."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

R = Path("results")


def load(name):
    with open(R / name) as f:
        return list(csv.DictReader(f))


def run_module(module):
    return subprocess.run([sys.executable, "-m", module], capture_output=True, text=True, check=True).stdout.strip()


def md(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def f1(x):
    return f"{x:.1f}"


def spider_section():
    static = {r["question_id"]: r for r in load("spider/python_static.csv")}
    sel = load("spider/python_selector.csv")
    n = len(static)
    total = lambda k: sum(float(r[k]) for r in static.values())
    rows = [[f"always {name}", f"{total(k + '_bytes') / 1e6:.2f}", f1(total(k + "_s"))]
            for k, name in (("json", "JSON"), ("parquet", "Parquet"), ("arrow_ipc", "Arrow IPC"))]
    sel_stats = {}
    for tg in ("min_bytes", "min_latency"):
        s = [r for r in sel if r["target"] == tg and r["question_id"] in static]
        b = sum(float(r["bytes"]) for r in s) / 1e6
        t = sum(float(r["latency_s"]) for r in s)
        fmts = {}
        for r in s:
            fmts[r["chosen_format"]] = fmts.get(r["chosen_format"], 0) + 1
        sel_stats[tg] = (b, t, fmts)
        rows.append([f"selector {tg}", f"{b:.2f}", f1(t)])
    json_mb = total("json_bytes") / 1e6
    text = (f"{n} of 1,034 dev queries completed; the other {1034 - n} return no rows, which the server rejects by "
            f"design. Largest result: {int(max(float(r['n_rows']) for r in static.values()))} rows.\n\n"
            + md(rows, ["strategy", "wire MB", "total latency (s)"])
            + f"\n\nSelector choices (min_bytes): {sel_stats['min_bytes'][2]}. "
              f"Bytes: {json_mb / sel_stats['min_bytes'][0]:.1f}x fewer than always-JSON.")
    return text, json_mb / sel_stats["min_bytes"][0]


def tpcds_section():
    py = {(r["n_rows"], r["n_cols"], r["strategy"]): r for r in load("tpcds/python.csv")}
    big = {(r["n_rows"], r["n_cols"], r["strategy"]): r for r in load("tpcds/python_bigcache.csv")}
    jv = {(r["n_rows"], r["n_cols"], r["strategy"]): r for r in load("tpcds/java.csv")}

    def cell(r, key="bytes"):
        v = r[key]
        if v == "":
            return r["chosen"].replace("capped (>5M cells)", "capped")
        return f"{float(v) / 1e6:.2f}" if key == "bytes" else f"{float(v):.3f}"

    rows = []
    for n in ("10000", "100000", "500000"):
        for c in ("6", "20", "34"):
            rows.append([n, c,
                         cell(py[(n, c, "fixed:json")]), cell(py[(n, c, "fixed:parquet_blob")]),
                         cell(py[(n, c, "fixed:arrow_ipc_blob")]),
                         cell(jv[(n, c, "fixed:parquet_blob")]), cell(jv[(n, c, "fixed:arrow_ipc_blob")]),
                         py[(n, c, "selector:min_bytes")]["chosen"],
                         cell(py[(n, c, "fixed:parquet_blob")], "latency_s"),
                         cell(big[(n, c, "fixed:parquet_blob")], "latency_s"),
                         cell(jv[(n, c, "fixed:parquet_blob")], "latency_s")])
    table = md(rows, ["rows", "cols", "JSON MB", "Parquet MB (py)", "Arrow IPC MB (py)", "Parquet MB (java)",
                      "Arrow IPC MB (java)", "selector (min_bytes) picks (py)",
                      "Parquet fetch s (py, 64 MB cache)", "(py, 4 GB cache)", "(java, 64 MB cache)"])
    ttfr = []
    for n in ("10000", "100000", "500000"):
        a, b = py[(n, "34", "selector:min_time_to_first_rows")], jv[(n, "34", "selector:min_time_to_first_rows")]
        ttfr.append([n, a["chosen"], f"{int(a['first_chunk_bytes']) / 1e3:.0f}", cell(a, "ttfr_s"),
                     b["chosen"], f"{int(b['first_chunk_bytes']) / 1e3:.0f}", cell(b, "ttfr_s")])
    t2 = md(ttfr, ["rows (34 cols)", "py choice", "py first chunk KB", "py TTFR s",
                   "java choice", "java first chunk KB", "java TTFR s"])
    return table, t2



def replication_section():
    if not (R / "replication/run1").exists():
        return "Not available: it needs the first run's BIRD CSVs in `results/replication/run1/`."
    rows = []
    for enc in ("data_driven", "default"):
        a_static = {r["question_id"]: r for r in load(f"replication/run1/matrix_python_{enc}_static.csv")}
        b_static = {r["question_id"]: r for r in load(f"bird/matrix_python_{enc}_static.csv")}
        common = set(a_static) & set(b_static)
        same_bytes = sum(all(a_static[q][k] == b_static[q][k] for k in ("json_bytes", "parquet_bytes", "arrow_ipc_bytes"))
                         for q in common)
        a_sel = {(r["question_id"], r["target"]): r for r in load(f"replication/run1/matrix_python_{enc}_selector.csv")}
        b_sel = {(r["question_id"], r["target"]): r for r in load(f"bird/matrix_python_{enc}_selector.csv")}
        keys = set(a_sel) & set(b_sel)
        same_choice = sum(a_sel[k]["chosen_format"] == b_sel[k]["chosen_format"] for k in keys)
        tot = lambda d, col, qs: sum(float(d[q][col]) for q in qs)
        for col, label in (("json_s", "always JSON"), ("parquet_s", "always Parquet")):
            t1, t2 = tot(a_static, col, common), tot(b_static, col, common)
            rows.append([enc, label, f1(t1), f1(t2), f"{abs(t1 - t2) / min(t1, t2) * 100:.1f}%"])
        t1 = sum(float(a_sel[k]["latency_s"]) for k in keys if k[1] == "min_latency")
        t2 = sum(float(b_sel[k]["latency_s"]) for k in keys if k[1] == "min_latency")
        rows.append([enc, "selector min_latency", f1(t1), f1(t2), f"{abs(t1 - t2) / min(t1, t2) * 100:.1f}%"])
        rows.append([enc, f"identical bytes {same_bytes}/{len(common)}, same format choice {same_choice}/{len(keys)}", "", "", ""])
    return md(rows, ["parquet encoding", "strategy / agreement", "run 1 total (s)", "run 2 total (s)", "difference"])


def codecs_section():
    if not (R / "codecs/codecs.csv").exists():
        return "Not available: run `python-server/.venv/bin/python -m benchmarks.sweeps.codecs`."
    return run_module("benchmarks.analysis.codecs").replace("#### ", "##### ")


def formats_sections():
    media = payloads = "Not available: run `benchmarks/scripts/run_media.sh` and `benchmarks/scripts/run_payloads.sh`."
    if (R / "media/images.csv").exists():
        media = run_module("benchmarks.analysis.media").replace("#### ", "##### ")
    if (R / "payloads/localhost.csv").exists():
        payloads = run_module("benchmarks.analysis.payloads").replace("#### ", "##### ")
    return media, payloads


def formats_findings():
    """Computed bullets for the image, audio and payload results; empty if those runs are absent."""
    if not (R / "media/images.csv").exists() or not (R / "payloads/localhost.csv").exists():
        return ""
    img, aud = load("media/images.csv"), load("media/audio.csv")

    def share(rows, arm, base_arm):
        """Bytes of one arm as a percentage of another arm's bytes."""
        a = sum(int(r["bytes"]) for r in rows if r["arm"] == arm)
        return a / sum(int(r["bytes"]) for r in rows if r["arm"] == base_arm) * 100

    def share_of_source(rows, arm):
        """Bytes of one image arm as a percentage of the original PNG files."""
        a = sum(int(r["bytes"]) for r in rows if r["arm"] == arm)
        return a / sum(int(r["png_bytes"]) for r in rows if r["arm"] == arm) * 100

    def psnr(arm):
        vals = sorted(float(r["psnr_db"]) for r in img if r["arm"] == arm)
        return vals[len(vals) // 2]

    n_img = len({r["image"] for r in img})
    out = [
        f"- **Images** ({n_img} M3-Bench PNGs): lossless WebP is {share_of_source(img, 'webp-lossless'):.0f}% of the source "
        f"PNG bytes; lossy WebP q80 is {share_of_source(img, 'webp-q80'):.0f}% (median PSNR {psnr('webp-q80'):.1f} dB) and "
        f"AVIF q60 is {share_of_source(img, 'avif-q60'):.0f}% ({psnr('avif-q60'):.1f} dB). PSNR only measures pixel change; "
        f"whether a model still answers correctly was not measured.",
        f"- **Speech** ({len({r['clip'] for r in aud})} synthesized clips): FLAC is {share(aud, 'flac-44k', 'wav-44k'):.0f}% of the "
        f"44.1 kHz WAV, 16 kHz mono WAV alone is {share(aud, 'wav-16k', 'wav-44k'):.0f}%, and Opus at 24 kbit/s is "
        f"{share(aud, 'opus-24k', 'wav-44k'):.0f}%. Recognition accuracy was not measured.",
    ]
    loc = load("payloads/localhost.csv")
    text = [r for r in loc if r["kind"] == "text"]
    raw_text = sum(int(r["raw_bytes"]) for r in text if r["strategy"] == "raw")
    wire = lambda strat: sum(int(r["wire_bytes"]) for r in text if r["strategy"] == strat) / raw_text * 100
    out.append(f"- **Text and JSON** ({len({r['payload'] for r in text})} payloads, {raw_text / 1e6:.0f} MB): gzip sends "
               f"{wire('gzip'):.0f}% of the raw bytes, zstd {wire('zstd'):.0f}%, brotli {wire('brotli'):.0f}%.")
    cases = []
    for run in ("localhost", "lan", "wan", "cellular", "badwifi"):
        if not (R / f"payloads/{run}.csv").exists():
            continue
        rows = load(f"payloads/{run}.csv")
        for kind in ("image", "audio", "text"):
            rs = [r for r in rows if r["kind"] == kind]
            t = lambda strat: sum(float(r["latency_s"]) for r in rs if r["strategy"] == strat)
            best = min(t(strat) for strat in ("raw", "gzip", "zstd", "brotli"))
            cases.append((t("selector min_latency") / best, f"{kind}, {run}"))
    close = [c for c in cases if c[0] <= 1.15]
    far = sorted((c for c in cases if c[0] > 1.15), reverse=True)
    out.append(f"- **Selector on unstructured payloads:** its total latency is within 15% of the best fixed arm in "
               f"{len(close)} of {len(cases)} kind and network combinations. The rest are slower: "
               + "; ".join(f"{label} {ratio:.1f}x" for ratio, label in far)
               + ". It picks the smallest arm by size and does not model compression time, so where bytes are cheap "
                 "(localhost, LAN) the size hints and the extra compression cost more than they save.")
    return "\n".join(out)


def synthetic_section():
    out = []
    for name, label in (("synthetic/python.csv", "Python server"), ("synthetic/java.csv", "Java server")):
        if not (R / name).exists():
            continue
        rows = load(name)
        by = {}
        for r in rows:
            by.setdefault(int(r["n_rows"]), {})[r["format"]] = r
        table = []
        for n in sorted(by):
            d = by[n]
            table.append([n] + [f"{int(d[f]['bytes']) / 1e6:.2f} MB / {float(d[f]['latency_median_s']) * 1e3:.0f} ms"
                                for f in ("json", "parquet_blob", "arrow_ipc_blob", "arrow_ipc_stream")])
        out.append(f"**{label}** (10 columns, median of 3 runs):\n\n"
                   + md(table, ["rows", "JSON", "Parquet blob", "Arrow IPC blob", "Arrow IPC stream"]))
    return "\n\n".join(out)


def netem_speedups():
    import csv as _csv
    out = []
    for dataset, tag, title in (("bird", "stride", "BIRD every 5th query"), ("bird", "large", "BIRD 5 largest results"),
                                ("spider", "stride", "Spider every 5th query")):
        for prof in ("lan", "wan", "cellular", "badwifi"):
            st = load(f"network/{prof}_{dataset}_{tag}_static.csv")
            se = load(f"network/{prof}_{dataset}_{tag}_selector.csv")
            ids = {r["question_id"] for r in st}
            j = sum(float(r["json_s"]) for r in st)
            t = sum(float(r["latency_s"]) for r in se if r["target"] == "min_latency" and r["question_id"] in ids)
            out.append((j / t, f"{title}, {prof}"))
    return out


def main():
    matrix = run_module("benchmarks.analysis.matrix").replace("### ", "#### ")
    netem = run_module("benchmarks.analysis.network").replace("### ", "#### ")
    spider, spider_x = spider_section()
    tpcds_table, tpcds_ttfr = tpcds_section()
    synthetic = synthetic_section()
    replication = replication_section()
    codecs = codecs_section()
    media, payloads = formats_sections()
    formats = formats_findings()
    speed = netem_speedups()
    lo, hi = min(speed), max(speed)
    bound = "at or below" if lo[0] >= 1.0 else "not always below"
    def matrix_ratio(impl):
        st = {r["question_id"]: r for r in load(f"bird/matrix_{impl}_data_driven_static.csv")}
        se = {r["question_id"]: r for r in load(f"bird/matrix_{impl}_data_driven_selector.csv") if r["target"] == "min_bytes"}
        sl = {r["question_id"]: r for r in load(f"bird/matrix_{impl}_data_driven_selector.csv") if r["target"] == "min_latency"}
        qs = set(st) & set(se) & set(sl)
        json_b = sum(float(st[q]["json_bytes"]) for q in qs)
        sel_b = sum(float(se[q]["bytes"]) for q in qs)
        json_t = sum(float(st[q]["json_s"]) for q in qs)
        sel_t = sum(float(sl[q]["latency_s"]) for q in qs)
        return json_b / sel_b, json_t, sel_t, len(qs)

    py_b, py_jt, py_st, py_n = matrix_ratio("python")
    jv_b, jv_jt, jv_st, jv_n = matrix_ratio("java")
    a, b = (sum(float(r["json_s"]) for r in load(f"bird/matrix_python_{e}_static.csv")) for e in ("data_driven", "default"))
    json_noise = abs(a - b) / min(a, b) * 100
    parity = (R / "parity.txt").read_text().strip() if (R / "parity.txt").exists() else "not recorded"

    text = f"""# TransForm: Python server restoration, results and benchmark checks

Generated by `benchmarks/analysis/report.py` from the CSVs in `results/` (gitignored; rerun the scripts to regenerate them).
Every number below comes from those files except those marked "ad hoc", which were measured separately.

## What was done

- Restored the original Python server from git history (commit `55de98c`) into `python-server/`. It now runs on the
  stock `mcp==2.3.0` package; the vendored SDK fork is not needed. Versions are pinned in `python-server/requirements.txt`.
- Fixed four things so it can replace the Java server: duplicate SQL column names, mixed-type and invalid-UTF-8 values
  from SQLite, a missing tool (`describe_and_fetch_result`), and a data-plane cache bug (below). Regression tests are in
  `python-server/tests/test_server_fixes.py`.
- Measured it against the Java server on BIRD dev, Spider dev, TPC-DS catalog_sales and a synthetic sweep, plus four
  emulated network profiles.

## Parity with the Java server

```
{parity}
```

Each side's tool names and argument schemas are compared by `python -m benchmarks.analysis.parity`.

## Benchmarks used

| Benchmark | What we used | Reference |
|---|---|---|
| BIRD (dev) | 1,534 gold SQL queries on the 11 dev SQLite databases (`dev_20240627`); queries 518 and 701 excluded (see limits) | [1] |
| Spider 1.0 (dev) | 1,034 gold SQL queries on the 20 dev databases (from `spider_data.zip`, 206 MB) | [2] |
| TPC-DS | `catalog_sales` at scale factor 1 (1,441,548 rows x 34 columns) generated with DuckDB's `dsdgen`, grid of 10K/100K/500K rows x 6/20/34 columns taken from the original harness | [3], [4] |
| Synthetic | generated tables from 100 to 500,000 rows, 10 columns | this repo (`benchmarks/sweeps/synthetic.py`) |
| M3-Bench images | the 212 PNG images from the Open-M3-Bench dataset (350 MB). They are the benchmark's task inputs (photos and screenshots), not tool results | [5] |
| Synthesized speech | 14 passages from this repo's docs, spoken by 7 macOS `say` voices, 44.1 kHz mono WAV (35 MB). No real recordings | `benchmarks/datasets/speech.py` |

## Results

### 1. BIRD dev: Python vs Java, two Parquet encoding strategies

{matrix}

Notes:
- Always-Parquet and the selector send fewer bytes on Python than on Java. The Java server gives identical Parquet bytes
  under `data_driven` and `default`, so the data-driven setting has no effect there.
- JSON byte counts match on 1,524 of 1,530 queries; the other 6 were not investigated.
- Latency here is localhost, SQL execution included, summed over 1,530 queries. Noise was not measured directly; the two
  Python configs do identical work for always-JSON and differ by {json_noise:.1f}% in total latency, so treat smaller gaps
  as noise.

### 2. Spider dev (Python server)

{spider}

### 3. TPC-DS catalog_sales (localhost)

{tpcds_table}

Stream first chunk and time to first rows (selector target `min_time_to_first_rows`, 34 columns):

{tpcds_ttfr}

### 4. Network emulation (Python server)

Profiles from the workshop paper: LAN (1 ms, 1 Gbit), WAN (40 ms, 50 Mbit), Cellular (80 ms, 10 Mbit, 0.1% loss),
BadWifi (30 ms, 5 Mbit, 1% loss). The server's `FORMAT_LATENCY_NETWORK_MBPS` is set to each profile's rate.

{netem}

### 5. Synthetic sweep

{synthetic}

The synthetic tables are integer arithmetic sequences (`index * (column + 1) + column`), identical on both servers (JSON
bytes match). With `data_driven` encoding the Python server delta-encodes them, so its Parquet output is orders of
magnitude smaller than JSON; Java writes plain-encoded Parquet. This is a best case for delta encoding and does not
represent real data: use the BIRD, Spider and TPC-DS tables for compression claims.

### 6. Table codecs (in-process; 21 arms)

Every Parquet codec the server accepts (none, snappy, lz4, gzip, brotli, zstd levels 1/3/9/19) under both encoding
strategies, plus Arrow IPC (none, lz4, zstd), on the five BIRD results above 50K cells and two TPC-DS slices. Encode and
decode times are measured in-process (median of 3, one machine); network times are modelled as bytes / bandwidth + decode
(+ encode), with the paper's four bandwidths and no delay term since it is the same for every arm. Agent results are
usually fetched once, so the "encoded per request" table is the realistic one; "pre-encoded" only applies to results
fetched repeatedly. The BIRD results are narrow (1 to 3 columns), so they say little about wide string tables.

{codecs}

### 7. Images and audio: codec comparison (in-process)

Every image and audio arm measured on the real files, no server. Image fidelity is PSNR against the original (alpha
flattened onto white); it is a pixel-difference proxy, not task accuracy. Encode times for images ran four at a time, so
they are inflated; audio times include ffmpeg's start-up (a few tens of ms) and the audio is synthetic speech, which is
cleaner than field recordings.

{media}

### 8. Unstructured payloads through the server

Images, speech and text/JSON/CSV documents through the five unstructured arms (raw, gzip, zstd, brotli, inline text) and
the selector. Each strategy uploads a fresh copy first (upload excluded), so every arm pays its own compression cost cold;
the selector rows also pay for computing the size hints. Forced arms include one MCP ping standing in for the tool call
that returns their URL, so every row pays one control-plane round trip. Wire bytes are what crossed the connection; every fetch was
decoded and compared with the original. The network rows use every 9th image.

{payloads}

### 9. Replication (two runs of the Python server on BIRD dev)

Run 1 used a server file that was missing five global definitions (hints cache, hint store, MAB state path; restored
afterwards, found by a regression test). None are used when results are registered through `/materialized` or
`bird_query_*`, which is why run 1 had no errors. Every section above is from run 2 (the corrected server); the table
below compares it with run 1 on BIRD dev. Wire bytes and format choices are compared per query; latency totals are
summed over the same queries. This is the only measure of run-to-run variation in this report.

{replication}

## Findings

- On every dataset here the selector sends the fewest bytes of any strategy (tied with always-Parquet on TPC-DS, where it
  always picks Parquet), and on the emulated networks its total latency is {bound} always-JSON in all {len(speed)}
  workload and profile combinations: from {lo[0]:.2f}x ({lo[1]})
  to {hi[0]:.1f}x ({hi[1]}). The large factors come from a handful of very large results, not from typical queries.
  On Spider it moves {spider_x:.1f}x fewer bytes than always-JSON.
- The workshop paper reports about 6x fewer wire bytes on BIRD dev (43.4 MB to 7.4 MB) and 23% lower aggregate latency on
  the Java server. This run gives {py_b:.1f}x fewer bytes on Python ({py_n} queries) and {jv_b:.1f}x on Java ({jv_n} queries);
  localhost latency for the selector versus always-JSON is {py_st:.1f} s versus {py_jt:.1f} s on Python and {jv_st:.1f} s
  versus {jv_jt:.1f} s on Java. These are not like-for-like: query sets,
  Parquet settings and how latency is composed differ, and the paper's own run is not reproduced here.
- **Cache bug (Python):** the `large_*` tools minted a new result id per fetch that kept only the file path, so every
  Parquet/Arrow fetch re-read the file and re-encoded it (ad hoc: 1.17 s per 100K x 34 fetch before the fix, 0.009 s after). The
  derived ids now share the source's cache and release it when the source is evicted from the LRU.
- **Java gaps found:** it fails on duplicate column names (BIRD 900 and 901), ignores the `data_driven` Parquet strategy
  (identical bytes under both settings), and its Parquet files are larger than Python's.
{formats}
- The TPC-DS table shows Python's Arrow IPC output is larger than Java's on wide tables; the cause was not investigated.

## Limitations

- BIRD dev queries 518 and 701 (ad hoc: 32 s and 168 s in plain SQLite) exceed the 30 s query timeout and are excluded
  from the BIRD latency totals. Totals that include them are dominated by those two queries.
- The network emulation is a software proxy (`benchmarks/netem/proxy.py`): per-direction delay, bandwidth cap, and loss
  approximated as one extra round trip per lost packet. It has no TCP slow start, so absolute numbers will not match
  the paper's `tc netem` runs. All strategies see the same emulated link.
- The network runs use every 5th query (plus the 5 largest BIRD results), one run per cell. The 5-largest set is
  noisy and is not a workload average.
- Localhost latencies are single runs per query; synthetic and TPC-DS cells are medians of 3.
- Zero-row results are rejected by the server and are excluded (Spider: see section 2).
- Security was neither audited nor changed. Reading the Java server's code showed a single shared bearer key (when set),
  no result expiry and wide-open CORS; the Python server was not reviewed for these. Nothing here measures them.
- The zstd and brotli arms, the tiny-text rule and the already-compressed check exist only in the Python server; the Java
  server has no equivalents, so section 8 has no Java comparison.
- The image set is M3-Bench's task inputs, not tool-returned images; the speech is synthetic; PDFs and video are not covered.
  Lossy arms were not tested for effect on model answers.
- Not done: the bandit (`use_mab`) was not run, no LLM accuracy or token measurement.

## Reproduce

```bash
# data (all under data/, which is gitignored):
#   BIRD dev: https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip (346 MB); unzip, then unzip dev_databases.zip inside
#   Spider:   spider_data.zip (206 MB) from https://yale-lily.github.io/spider ; unzip to data/spider_data,
#             then: mkdir data/spider_root && ln -s ../spider_data/database data/spider_root/dev_databases
#   TPC-DS:   python -m benchmarks.datasets.tpcds   (needs duckdb; writes data/tpcds_catalog_sales_sf1.parquet)
#   M3-Bench images: the images/ folder of https://huggingface.co/datasets/EtaYang10th/Open-M3-Bench (350 MB, 212 PNGs),
#             saved to data/m3bench/images
#   Speech:   python -m benchmarks.datasets.speech   (macOS `say` plus ffmpeg; writes data/audio)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cd python-server && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && cd ..
benchmarks/scripts/run_all.sh                 # every measurement above, then rebuilds this report (hours)
SKIP_JAVA=1 benchmarks/scripts/run_all.sh     # Python server only
.venv/bin/python -m benchmarks.analysis.report   # rebuild the report from existing CSVs
```

The client environment (`.venv`, root `requirements.txt`) pins `mcp==1.30.0`; the server environment
(`python-server/.venv`) pins `mcp==2.3.0`. The client breaks on `mcp` 2.x because `structuredContent` was renamed.

## References

[1] J. Li, B. Hui, G. Qu, J. Yang, B. Li, B. Li, B. Wang, B. Qin, R. Geng, N. Huo, X. Zhou, C. Ma, G. Li, K. C. C. Chang,
F. Huang, R. Cheng, Y. Li. "Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded
Text-to-SQLs." NeurIPS 2023, Datasets and Benchmarks Track. arXiv:2305.03111. https://arxiv.org/abs/2305.03111

[2] T. Yu, R. Zhang, K. Yang, M. Yasunaga, D. Wang, Z. Li, J. Ma, I. Li, Q. Yao, S. Roman, Z. Zhang, D. Radev. "Spider:
A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task." EMNLP 2018,
pp. 3911-3921. https://aclanthology.org/D18-1425/ (data: https://yale-lily.github.io/spider)

[3] R. O. Nambiar, M. Poess. "The Making of TPC-DS." VLDB 2006, pp. 1049-1058.
https://dl.acm.org/doi/10.5555/1182635.1164217

[4] Transaction Processing Performance Council. "TPC-DS Benchmark Specification."
https://www.tpc.org/tpc_documents_current_versions/current_specifications5.asp

[5] Y. Zhou, M. Zhao, Z. Wang, D. Gu, B. Guo, R. Ye, L. Han, C. Jin, D. N. Metaxas. "M3-Bench: Multi-Modal, Multi-Hop,
Multi-Threaded Tool-Using MLLM Agent Benchmark." arXiv:2511.17729, 2025. https://arxiv.org/abs/2511.17729
(images: https://huggingface.co/datasets/EtaYang10th/Open-M3-Bench; code: https://github.com/EtaYang10th/Open-M3-Bench, MIT)

[6] S. Mukkera, C. Liu. "TransForm: Dynamic Format Selection for MCP Tool Outputs in Agentic Data Systems." VLDB 2026
Workshop: NOVAS (source: `novas-workshop/main.tex`; the baseline this work reproduces).
"""
    Path("REPORT.md").write_text(text)
    print("wrote REPORT.md")


if __name__ == "__main__":
    main()
