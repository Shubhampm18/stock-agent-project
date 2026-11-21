# Demo - the point-in-time guarantee

Same date, two answers: what a look-ahead leak actually costs.

Run from `Source-Code/indian-stocks-mcp`:

    uv sync
    uv run python demo/vintage_drift.py

The script prints a table comparing the price **as it could be known on the
as-of date** (only corporate actions with ex-date <= as_of) against the
vendor's **today-adjusted** price for the same day, and writes
`docs/media/vintage_drift.svg` (also copied to `Screenshots/`).

Cases chosen to straddle real corporate actions:
RELIANCE 2024-10-01 (1:1 bonus 2024-10-28), HDFCBANK 2025-08-01 (1:1 bonus),
TCS 2018-05-01 (bonus), INFY 2020-01-01 (dividends only).

A naive backtest reading today's adjusted prices silently absorbs this drift;
the server's `as_of` guarantee removes it.
