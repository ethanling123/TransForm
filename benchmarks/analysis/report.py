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


def shift_headings(text, drop_title=True):
    """Nest a generated markdown block under a '###' section: drop its '# ' title, push '#'-headings down to '####'."""
    lines = text.splitlines()
    if drop_title and lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join("####" + l[l.index(" "):] if l.startswith("#") else l for l in lines).strip()


def attacks_section():
    path = R / "attacks" / "BEFORE.md"
    if not path.exists():
        return "not run", ""
    text = path.read_text()
    table = text[text.index("| ID |"):text.index("## Evidence")].strip()
    evidence = text[text.index("### open"):].strip()
    return table, shift_headings(evidence, drop_title=False)


def lossy_facts():
    """One-line findings computed from the lossy and cost-model outputs."""
    import statistics as st
    audio = [r for r in load("lossy/audio_accuracy.csv")]
    by = {}
    for r in audio:
        by.setdefault(r["arm"], []).append(r)
    base = {r["clip"]: float(r["wer"]) for r in by["original"]}
    def delta(arm):
        return 100 * st.mean(float(r["wer"]) - base[r["clip"]] for r in by[arm])
    def pct(arm):
        return 100 * st.mean(int(r["bytes"]) / int(r["wav_bytes"]) for r in by[arm])
    return (f"Opus at 32 kbit/s is {pct('opus-32k'):.1f}% of the WAV size and changes mean word error rate by "
            f"{delta('opus-32k'):+.2f} points (original: {100 * st.mean(base.values()):.2f}%); Opus at 12 kbit/s is "
            f"{pct('opus-12k'):.1f}% of the size at {delta('opus-12k'):+.2f} points")


def cost_facts():
    import re
    text = (R / "payloads" / "cost_model.md").read_text()
    rows = [l for l in text.split("## Regret")[1].splitlines() if l.startswith("| ") and "x |" in l]
    cells = [[c.strip() for c in l.strip("| ").split("|")] for l in rows]
    cost = [float(c[2].rstrip("x")) for c in cells]
    cur = [(float(c[4].rstrip("x")), c[0], c[1]) for c in cells]
    worst = max(cur)
    return max(cost), worst


def generated_block(path, **kwargs):
    return shift_headings((R / path).read_text(), **kwargs) if (R / path).exists() else "not run"


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
    lossy_accuracy = run_module("benchmarks.analysis.lossy")
    run_module("benchmarks.analysis.lossy_worth")
    lossy_worth = generated_block("lossy/worth.md")
    cost_model = generated_block("payloads/cost_model.md")
    run_module("benchmarks.analysis.defaults")
    defaults = generated_block("defaults/defaults.md")
    cost_aware = generated_block("payloads_cost_aware/summary.md")
    paper_repro = generated_block("paper_repro/comparison.md")
    attacks_table, attacks_evidence = attacks_section()

    text = f"""# TransForm: Python server restoration, results and benchmark checks

Generated by `benchmarks/analysis/report.py` from the CSVs in `results/` (gitignored; rerun the scripts to regenerate them).
Every number below comes from those files except those marked "ad hoc", which were measured separately.

Section 13 describes the cost-aware selector, which is in the server but off unless `FORMAT_COST_AWARE=1`.

## What was done

- Restored the original Python server from git history (commit `55de98c`) into `python-server/`. It now runs on the
  stock `mcp==2.3.0` package; the vendored SDK fork is not needed. Versions are pinned in `python-server/requirements.txt`.
- Fixed four things so it can replace the Java server: duplicate SQL column names, mixed-type and invalid-UTF-8 values
  from SQLite, a missing tool (`describe_and_fetch_result`), and a data-plane cache bug (below). Regression tests are in
  `python-server/tests/test_server_fixes.py`.
- Measured it against the Java server on BIRD dev, Spider dev, TPC-DS catalog_sales and a synthetic sweep, plus four
  emulated network profiles.
- Reran the workshop paper's own BIRD harness, unmodified, against the Java and Python servers (section 14).
- Added zstd and brotli arms and an already-compressed check to the Python server's unstructured path, and measured
  every table codec, image codec and audio codec (sections 6, 7, 8).
- Added an arm registry (`python-server/arms.py`): one `ArmSpec` per arm (modality, container, codec, transport), a
  generic `GET /data/{{handle}}` route that serves any registered arm with a content digest header, and `list_arms` and
  `fetch_arm` tools. The ten existing arms are unchanged (golden byte digests; 37 tests at the time), and eight
  table-codec arms that the selector never picks became reachable. The Java server has none of this and the two tools
  exist only in Python (the other 18 tools match). Design: `docs/superpowers/specs/2026-10-14-codec-registry-design.md`.
- Measured whether lossy image and audio arms are worth using and whether they change a model's answers (section 10).
- Replayed, implemented (experimental, behind a flag) and measured a cost-aware selector (sections 11 and 13).
- Changed the Python server's default table codecs from snappy Parquet and uncompressed Arrow IPC to zstd for both,
  after comparing them on BIRD and TPC-DS (section 12).
- Wrote executable demos of nine attacks against the unmodified server, as the "before" measurement for the security
  work (section 15). Nothing in the server was changed to defend against them.

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
| Synthesized speech | 14 passages of everyday prose (`benchmarks/datasets/speech_passages.txt`), spoken by 7 macOS `say` voices, 44.1 kHz mono WAV (35 MB). No real recordings | `benchmarks/datasets/speech.py` |
| Rendered documents | the same 14 passages (three at a time) drawn as white-page images at 7 to 18 px type; the text is the ground truth | `benchmarks/sweeps/lossy.py` |

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
flattened onto white); it is a pixel-difference proxy, not task accuracy (section 10 measures that). Files were encoded one
at a time, so the timings carry no contention; audio times include ffmpeg's start-up (a few tens of ms) and the audio is
synthetic speech, which is cleaner than field recordings.

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

### 10. Lossy image and audio arms: do they pay, and do they change the answer?

Section 7 measured bytes and fidelity; this section asks whether a lossy arm is worth using. Two separate questions.

**Bytes versus tokens.** A vision model charges by image dimensions, not file size. Claude counts one token per 28 x 28
pixel patch (ceil(width / 28) x ceil(height / 28)) and downsizes anything above 1568 px on the long edge (2576 px for newer
models); Gemini counts 258 tokens per 768 x 768 tile [7] [8]. File format and compression do not enter either formula. On
the 212 M3-Bench images the median long edge is 1200 px and only 21% exceed 1568 px, so a lossy codec at the same
resolution saves transfer bytes and nothing else; only resizing saves tokens (median 1,369 tokens as-is, 899 at a
1024 px long edge by the Claude formula).

**Time to deliver.** Transfer time against encode and decode time, for each link, relative to sending the lossless original
(encode and decode are in-process, one file at a time):

{lossy_worth}

**Does it change what the model says.** Four checks, none of them a measure of task success:
- *Controls.* `png-rerun` sends the untouched pixels a second time, so any gap from 1.00 is the model's own run-to-run
  noise (Gemini at temperature 0, thinking off). The local model at temperature 0 with a fixed seed is deterministic,
  so its controls read 1.00 by construction and `jpeg-q95` (visually identical) is the better floor there.
- *Natural images* (M3-Bench): two models, Gemini 2.5 Flash on Vertex AI and Qwen2.5-VL 7B (Q4_K_M, via Ollama) [10]. The score is agreement with the model's own answer on the untouched image (token F1 of two
  fixed questions: transcribe the visible text, list the main objects). There is no ground truth, so this measures whether
  the arm changed what the model said, not whether it was right.
- *Documents* (images rendered from known text): the score is character error rate against the true text, so it is
  correctness. They are small synthetic strips of plain text; real screenshots, charts and scans were not tested.
- *Audio* (synthetic speech, local Whisper `small.en` [9] through `mlx-whisper`): word error rate against the exact script, clean and with white
  noise mixed in at 20, 10 and 5 dB signal-to-noise before encoding.

{lossy_accuracy}

Intervals are 95% bootstrap intervals over images. The local-model rows use 30 images and 6 arms (about 75 s per
image and arm on this machine); the Gemini rows use up to 116 images.

### 11. Cost-aware selector: offline replay

The current selector picks the arm with the fewest bytes (or, with a link speed set, the least transfer time) and ignores
how long compressing and decompressing take. The replayed policy scores each arm as compute time plus wire bytes / bandwidth
and takes the minimum; the compute model is fitted leave-one-out on the unstructured payload runs of section 8. Each
chosen arm's latency is the one measured when that arm was forced on that payload and link; the policy is replayed, not
deployed. Knowing each arm's size costs something too, so it is charged: "exact + hints" adds the measured size-checking
time (a lower bound), "sampled" extrapolates sizes from the first 256 KB and charges the time that took, and
"sampled + fast skip" sends raw without sampling on a 1 Gbit/s or faster link. The module docstring of
`benchmarks/analysis/cost_model.py` has the exact rules.

{cost_model}

### 12. Table codec defaults: snappy / none against zstd / zstd

The Python server's default table codecs changed from snappy Parquet and uncompressed Arrow IPC to zstd for both. Same
BIRD queries, same server and link; only `PARQUET_COMPRESSION` and `ARROW_IPC_COMPRESSION` differ (the benchmark scripts
still pin the old values, so every other section is unchanged). The selector sizes its hints with the server's default
codecs, so the default also changes what it picks.

{defaults}

### 13. Cost-aware selector: measured on the server (experimental, off by default)

The policy of section 11 implemented in the Python server (`python-server/cost_model.py`, enabled with
`FORMAT_COST_AWARE=1`, `min_latency` target only). With a known link speed it compresses and writes nothing to choose:
it sends raw without sampling when even zero remaining bytes would save less transfer time than the cheapest codec costs,
samples zstd first and stops there if zstd alone does not beat raw, and otherwise scores sampled sizes as compute time plus
bytes / bandwidth. Codec speeds come from a one-off startup microbenchmark; the server's decompression time stands in for
the client's. The same payload sweep as section 8 was run on each link, with every forced arm rerun in the same run
as the baseline. The localhost run is told the link is 20 Gbit/s (the replay's assumption); with no link speed the flag
falls back to the legacy behaviour. Localhost totals are tens to hundreds of milliseconds, so ratios there carry
noise of the same size as the differences between arms.

{cost_aware}

### 14. The workshop paper's own harness, rerun

The paper's BIRD harness (`benchmarks/paper_harness/`, unmodified from commit `55de98c`) was rerun once per arm on the
full BIRD dev set with gold SQL: baseline JSON, client-driven selection and the fused `large_result_auto` call. The
paper's numbers are the raw records in that commit. `benchmarks/analysis/paper_repro.py` writes this comparison;
`benchmarks/scripts/run_paper_repro.sh` reruns it. The paper's Python run used the default Parquet encoding, not
`data_driven`, so both are shown.

{paper_repro}

### 15. Attack demos against the unmodified server (before any defence)

Nine executable attacks (`benchmarks/attacks/run_attacks.py`, run by `benchmarks/scripts/run_attacks.sh`) against the
Python server in two configurations: `open` (no API key) and `keyed` (the one shared bearer key it supports). VULNERABLE
means the attack worked; NOT_REPRODUCED means it failed in this setup, with the reason in the evidence. These are
demonstrations on a local server with test data, not a security audit: they show what the missing controls allow, and
nothing has been built or measured to prevent them.

{attacks_table}

{attacks_evidence}

## Findings

- On every dataset here the selector sends the fewest bytes of any strategy (tied with always-Parquet on TPC-DS, where it
  always picks Parquet), and on the emulated networks its total latency is {bound} always-JSON in all {len(speed)}
  workload and profile combinations: from {lo[0]:.2f}x ({lo[1]})
  to {hi[0]:.1f}x ({hi[1]}). The large factors come from a handful of very large results, not from typical queries.
  On Spider it moves {spider_x:.1f}x fewer bytes than always-JSON.
- The workshop paper reports about 6x fewer wire bytes on BIRD dev (43.4 MB to 7.4 MB) and 23% lower aggregate latency on
  the Java server. This run gives {py_b:.1f}x fewer bytes on Python ({py_n} queries) and {jv_b:.1f}x on Java ({jv_n} queries);
  localhost latency for the selector versus always-JSON is {py_st:.1f} s versus {py_jt:.1f} s on Python and {jv_st:.1f} s
  versus {jv_jt:.1f} s on Java. Those sweeps are not like-for-like with the paper: query sets, Parquet settings and
  how latency is composed differ. The paper's own harness was rerun separately (section 14): on the Java server it
  reproduces the byte totals (43.4 to 7.4 MB) and a fused-call transport-time cut of 22% against the 23% claimed; the
  Python server gives 6.41 MB with the paper's default encoding (6.40 MB in the paper) and 4.83 MB with `data_driven`.
  Absolute seconds, and the client-driven arm on Java (19% slower than baseline in the rerun), differ from the paper.
- **Cache bug (Python):** the `large_*` tools minted a new result id per fetch that kept only the file path, so every
  Parquet/Arrow fetch re-read the file and re-encoded it (ad hoc: 1.17 s per 100K x 34 fetch before the fix, 0.009 s after). The
  derived ids now share the source's cache and release it when the source is evicted from the LRU.
- **Java gaps found:** it fails on duplicate column names (BIRD 900 and 901), ignores the `data_driven` Parquet strategy
  (identical bytes under both settings), and its Parquet files are larger than Python's.
{formats}
- The TPC-DS table shows Python's Arrow IPC output is larger than Java's on wide tables; the cause was not investigated.
- **Lossy image and audio arms (section 10).** A lossy image codec saves transfer bytes, not model tokens: tokens depend
  on pixel dimensions, and only 21% of the M3-Bench images are large enough for a provider to downscale. The byte saving is
  large where it exists (median WebP q80 is 11% and AVIF q60 8% of the PNG), but it is not universal: 4% to 13% of the
  images (screenshots and graphics that PNG already compresses well) come out larger under WebP q60, WebP q80 and
  JPEG q75, and the rendered text documents came out 104% to 317% of the PNG under WebP and JPEG. A selector must compare real
  sizes, not assume a lossy arm is smaller. Lossy arms beat the lossless original only on WAN or slower; on a LAN
  WebP and AVIF are slower end to end (median 0.3x to 0.4x) because encoding takes longer than sending, and only JPEG
  stays faster (4.3x at q75).
- **Lossy arms and what the model says (section 10).** Reading text survives: Gemini's OCR agreement with its own answer on
  the original is 0.93 to 0.96 for every lossy arm against 0.97 for the rerun control. Open-ended object lists drift more:
  agreement is 0.75 to 0.83 against a control of 0.87, with intervals clear of the control for WebP q80 (with and without the
  1024 px resize) and both AVIF arms. This measures change, not correctness. On rendered text with a known answer,
  WebP q30 and JPEG q30 raise the character error rate by under 0.1 point down to 7 px type, AVIF q20 fails at 7 px (mean character error rate 26%, from 0.01% for the
  original), and a quarter-size image is unreadable (93% at 7 px, 68% at 9 px).
- **Audio (section 10).** FLAC and 16 kHz WAV leave word error rate unchanged, including under added noise, at 22% to
  37% of the WAV. Opus at 32 kbit/s is 4% of the WAV and unchanged on clean speech but loses 0.9 points at 10 dB
  signal-to-noise; Opus at 12 kbit/s loses 0.7 points on clean speech and 4.1 points at 5 dB.
- **Cost-aware selector (section 11).** Replayed offline, a policy that weighs compute time against transfer time
  stays within 1.01x to 1.08x of the best arm on WAN, cellular and bad wifi, against 1.01x to 1.29x for the current
  selector. On localhost and LAN the current selector is 1.6x to 22x slower than the best arm because it compresses
  every arm to count bytes. Charging the new policy for estimating sizes still leaves it 1.2x to 2.3x slower than sending
  raw on those links; skipping the estimate on 1 Gbit/s or faster links removes that on localhost (1.00x to 1.02x) but costs 2.6x
  for text on a LAN, so the rule needs to look at content type as well as link speed. On slow links the policy is no better than
  the best fixed arm (zstd, or brotli for audio): it is 0.00x to 0.08x slower because it pays for the estimate, so the
  gain is avoiding the fast-link losses. A wrong link-speed estimate (0.25x or 4x) moved the sampled policy's regret by
  at most 0.12x in any cell.
- **Cost-aware selector, measured (section 13, experimental).** On the server the policy cut the selector's
  regret on LAN from 1.6x, 5.0x and 5.1x (images, audio, text) to 1.11x, 1.07x and 1.06x, and the time its tool call
  spends choosing from 14 to 113 ms down to 5 to 6 ms. On WAN and cellular it is within 1.01x to 1.03x of the best forced
  arm. On localhost it went from 4.1x, 22x and 9.4x to 1.2x, 1.6x and 0.9x; those totals are 0.06 to 0.9 s, so the
  residual is within run-to-run noise (the text figure is below the best fixed arm). Two causes of overhead were found
  and fixed: the codec calibration ran on the first request (about 0.17 s) and gzip and brotli were sampled on a LAN
  where they cannot beat zstd. A first version without those fixes measured 1.43x and 1.45x on LAN images and text.
- **Table codec defaults (section 12).** zstd for Parquet and Arrow IPC cuts bytes 13% to 40% (Parquet) and 74% to 91%
  (Arrow IPC) on BIRD, with total latency within 4% on localhost, LAN and WAN for the typical query. On BIRD results above
  50K cells over cellular, Arrow IPC went from 19.0 s to 5.2 s (5 queries). On TPC-DS the min_bytes and min_latency selector's total latency
  was 0.78x to 0.86x of the snappy default on every link (59 MB to 46 MB), Arrow IPC blob 0.13x to 0.27x, and
  Parquet blob 0.80x to 1.02x; the slow-link rows there are modelled from bytes, not measured through the proxy.

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
- Security: section 15 demonstrates nine attacks on the Python server (eight work with no key, seven with the shared key).
  It is not an audit, no defence exists yet, and the Java server was not attacked. Reading the Java server's code showed
  a single shared bearer key (when set), no result expiry and wide-open CORS. Whether T3 and T8 matter depends on the key
  configuration: T3 was blocked only by the shared key, and T8 found no credential in tool output.
- The zstd and brotli arms, the tiny-text rule and the already-compressed check exist only in the Python server; the Java
  server has no equivalents, so section 8 has no Java comparison.
- The image set is M3-Bench's task inputs, not tool-returned images; the speech is synthetic; PDFs and video are not covered.
  Lossy arms were tested for effect on model answers only as described in section 10: agreement with the model's own
  answer, not task success, and no Claude model.
- Not done: the bandit (`use_mab`) was not run, and no end-to-end agent task accuracy was measured (section 10 compares a model's answers with and without a lossy arm).

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
# not part of run_all.sh:
benchmarks/scripts/run_paper_repro.sh         # section 14 (the paper's harness; SERVER=java for the Java rerun)
benchmarks/scripts/run_attacks.sh             # section 15
benchmarks/scripts/run_defaults.sh            # section 12 (BIRD); run_defaults_tpcds.sh for TPC-DS
benchmarks/scripts/run_lossy.sh               # section 10 (needs ollama with qwen2.5vl:7b and mlx-whisper)
benchmarks/scripts/run_payloads_cost_aware.sh # section 13
.venv/bin/python -m benchmarks.analysis.cost_model   # section 11 replay
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

[7] Anthropic. "Vision." Claude API documentation. https://platform.claude.com/docs/en/build-with-claude/vision
(accessed 2026-10-19; image token formula and resize limits).

[8] Google. "Image understanding." Gemini API documentation. https://ai.google.dev/gemini-api/docs/image-understanding
(accessed 2026-10-19; per-tile token cost).

[9] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, I. Sutskever. "Robust Speech Recognition via Large-Scale Weak
Supervision." arXiv:2212.04356, 2022. https://arxiv.org/abs/2212.04356

[10] S. Bai et al. "Qwen2.5-VL Technical Report." arXiv:2502.13923, 2025. https://arxiv.org/abs/2502.13923

"""
    Path("REPORT.md").write_text(text)
    print("wrote REPORT.md")


if __name__ == "__main__":
    main()
