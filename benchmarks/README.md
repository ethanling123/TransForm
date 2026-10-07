# Benchmarks

Scripts that measure how the TransForm servers move tool results (bytes on the wire and latency) and build
[`../REPORT.md`](../REPORT.md) from the output. Everything here is a client of the running server; the servers live in
[`../python-server/`](../python-server) and [`../java-server/`](../java-server).

## Layout

| Folder | What it holds |
|---|---|
| `sweeps/` | The measurement scripts. `bird.py` forces each format in turn; `selector.py` lets the server choose; both work for BIRD and Spider (`--dev`, `--sql-field`). `tpcds.py` runs the TPC-DS grid, `synthetic.py` the generated tables, `codecs.py` every table codec in-process (run it with `python-server/.venv`), `media.py` image and audio codecs in-process, `payloads.py` raw/gzip/zstd/brotli/inline arms and the selector on images, speech and text through the server. |
| `netem/` | `proxy.py`: a software network emulator (delay, bandwidth cap, loss) placed between client and server. |
| `analysis/` | `matrix.py`, `network.py`, `codecs.py`, `media.py` and `payloads.py` turn CSVs into tables, `parity.py` diffs two servers' tool lists, `report.py` writes `REPORT.md`. |
| `datasets/` | `tpcds.py`: generates TPC-DS `catalog_sales` with DuckDB. `speech.py`: synthesizes the speech clips (macOS `say` and ffmpeg). |
| `scripts/` | Shell drivers. `run_all.sh` reproduces everything; `run_matrix.sh`, `run_dataset.sh` and `run_network.sh` are its steps; `common.sh` starts and stops servers. |

## Where things go

- `data/` (gitignored): BIRD, Spider, TPC-DS, M3-Bench images (`data/m3bench/`) and speech (`data/audio/`). Download and generation steps are in the report's Reproduce section.
- `results/` (gitignored), one folder per benchmark: `bird/`, `spider/`, `tpcds/`, `synthetic/`, `network/`, `codecs/`, `media/`, `payloads/`, plus
  `replication/` (an earlier BIRD run used for the run-to-run comparison), `logs/` (server and sweep logs) and
  `archive/` (superseded early runs). `results/parity.txt` is the last tool-parity check.

## Run

Run from the repo root, with the client environment (`.venv`, from `requirements.txt`) and the server environment
(`python-server/.venv`) set up:

```bash
benchmarks/scripts/run_all.sh                   # everything, then rebuild REPORT.md (hours)
SKIP_JAVA=1 benchmarks/scripts/run_all.sh       # Python server only
IMPLS=python benchmarks/scripts/run_matrix.sh   # just the BIRD comparison
python -m benchmarks.analysis.report            # rebuild REPORT.md from existing CSVs
```

A single sweep, against a server that is already running on port 8000:

```bash
.venv/bin/python -m benchmarks.sweeps.bird --limit 50 --out /tmp/bird_static.csv
.venv/bin/python -m benchmarks.sweeps.selector --limit 50 --out /tmp/bird_selector.csv
```
