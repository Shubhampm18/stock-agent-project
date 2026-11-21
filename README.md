# stock agent - point-in-time NSE/BSE market data

Semester project. An MCP (Model Context Protocol) server that exposes 7
market-data tools for NSE/BSE equities. Every tool takes `as_of=YYYY-MM-DD`
and one shared function guarantees that no data after that date is ever
returned. A future date is rejected with an error, not silently clamped.

## Architecture

The server runs as a child process and communicates over stdio using the MCP
protocol (JSON-RPC 2.0). There is no web server, no port and no auth: the
client starts the process and owns its lifecycle.

```
client (dashboard / stock agent)              server (this package)
  |                                             |
  |  tools/list  -> tool schemas                |  FastMCP, 7 tools
  |  tools/call  -> result or error   <------> |  (@mcp.tool decorators)
  |                                             |
  LLM + tool-calling loop                       data.py
  SQLite memory (watchlist,                     - fetch_history / fetch_actions
  history, preferences)                         - enforce_as_of   <- the cutoff
                                                - rebuild_adjusted_with_actions
                                                yfinance underneath
```

How the point-in-time guarantee works:

1. Every tool calls the same helper `_get_frame()` in `data.py`.
2. `_get_frame()` calls `enforce_as_of(df, as_of)`, which slices the price
   history to rows with date <= as_of. Nothing later is ever visible to a tool.
3. If the requested `as_of` is in the future, the server raises
   `FutureDateError` instead of clamping to today.
4. `rebuild_adjusted_with_actions()` rebuilds the adjusted close using only
   corporate actions (splits, dividends) whose ex-date is <= as_of. So the
   same question about the same date gives the same answer no matter when
   it is asked. The vendor's own "Adj Close" changes retroactively every
   time a new dividend lands, which is why we rebuild it ourselves.
5. Every response carries provenance: `as_of`, `data_through` (date of the
   last row actually used) and `rows_used` (how many rows the cutoff admitted).

The agent on the client side is one LLM with a tool-calling loop over these
tools, plus a small SQLite memory (watchlist, query history, preferences).
Memory is used as context for the conversation; it is never used as a source
for a number.

## Technologies

- Python 3.12
- FastMCP (`mcp` package, 1.12.4) for the MCP server and stdio transport
- yfinance for NSE/BSE price data
- pandas for data handling
- sqlite3 (stdlib) for the agent's memory
- openai-compatible chat API for the agent's LLM (tool calling)
- Streamlit for the dashboard
- uv for environment and packaging; published as a wheel on GitHub Releases
- pytest for tests, run in CI (GitHub Actions)

## Execution instructions

```bash
# from Source-Code/indian-stocks-mcp/
uv sync
uv run python -m stock_analyst_mcp.server     # start the MCP server (stdio)

# run the tests (41 offline, 3 need network)
uv run pytest -q

# run the look-ahead demo: same date, two answers
uv run python demo/vintage_drift.py
```

Install from the published release instead of the source:

```bash
uvx --from https://github.com/Shubhampm18/indian-stocks-mcp/releases/download/v0.3.0/indian_stocks_mcp-0.3.0-py3-none-any.whl indian-stocks-mcp
```

## Tools

All 7 tools take `as_of` and return `data_through` and `rows_used`:

get_ohlcv, get_returns, get_technical_indicators, get_chart_patterns,
get_risk_metrics, get_index_history, get_peer_comparison

## Repository structure

```
Project-Repository/
├── README.md
├── Project-Report.pdf
├── Source-Code/indian-stocks-mcp/
├── Documentation/
│   ├── Project-Presentation.pdf
│   └── README.md
├── Screenshots/
└── Demo/
```

## Related repositories

- Server (published, v0.3.0): https://github.com/Shubhampm18/indian-stocks-mcp
- Agent and dashboard: https://github.com/Shubhampm18/indian-stocks-mcp-apps

This repository is the submission snapshot; the repos above are kept up to date.
