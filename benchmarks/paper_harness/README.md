# Workshop paper harness

`bench_bird_e2e.py`, `bench_nl2sql_materialized.py` and `summarize_bird_e2e.py` are copied unmodified from commit
`55de98c`, the harness behind the workshop paper's BIRD numbers. They import `format_selector` and `client`, so run them
through `benchmarks/scripts/run_paper_repro.sh`, which sets `PYTHONPATH`, starts the server and runs each arm.
`benchmarks/analysis/paper_repro.py` compares the output with the paper's raw records in git history.
