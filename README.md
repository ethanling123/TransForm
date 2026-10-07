# TransForm: Dynamic Format Selection for MCP Tool Outputs

Research artifact for the NOVAS workshop paper: a control/data-plane split for MCP tabular tool results with dynamic format selection (JSON, Parquet, Arrow IPC).

Artifact: https://github.com/shashank524/TransForm

## Layout

| Path | Description |
|------|-------------|
| `novas-workshop/` | Workshop paper (`main.tex`, `ref.bib`, figures) |
| `python-server/` | Python MCP server (restored from the original implementation; see its README) |
| `java-server/` | Java (Spring Boot) MCP server with fused format selection |
| `java-sdk/` | Vendored [MCP Java SDK](https://github.com/modelcontextprotocol/java-sdk) |
| `client/` | Python MCP client library (`mcp_client.py`, `llm_client.py`) |
| `benchmarks/` | Benchmark sweeps, network emulation, analysis, report builder and run scripts (see its README) |
| `REPORT.md` | Results and benchmark checks for the Python server (BIRD, Spider, TPC-DS, network profiles) |

## Java server

```bash
mvn -f pom.xml package -DskipTests -pl java-server -am
java --add-opens=java.base/java.nio=ALL-UNNAMED \
  -jar java-server/target/multimodal-mcp-0.1.0-SNAPSHOT.jar
```

MCP endpoint: `http://localhost:8000/mcp/mcp`

Optional BIRD SQLite: set `BIRD_SQLITE_ROOT` to the dev database directory.

## Python client

```bash
pip install -r requirements.txt
export MCP_URL=http://localhost:8000/mcp/mcp
export SERVER_URL=http://localhost:8000
python -m client.mcp_client
```

## Paper

```bash
cd novas-workshop && latexmk -pdf main.tex
```

Figures are in `novas-workshop/figures/`. Regenerate selection diagrams with `figures/render_selection_figures.sh` (requires [mermaid-cli](https://github.com/mermaid-js/mermaid-cli)).
