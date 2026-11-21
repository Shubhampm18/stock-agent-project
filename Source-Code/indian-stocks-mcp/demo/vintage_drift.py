"""Same date, two answers — what a look-ahead leak actually costs.

A naive backtest asks a vendor for adjusted prices *today* and uses them to
judge a date in the past. But an adjusted price is not a fact about that date:
it is that date's close multiplied by every split and dividend that has
happened SINCE. So the number you read for 2024-10-01 changes every time a
corporate action lands.

This script measures that drift with the server's own data layer:

    vintage(d)  = rebuild_adjusted_with_actions(raw, actions, as_of=d)
                  -> the price as it could be known on day d (actions <= d only)
    today(d)    = the vendor's AdjClose for day d
                  -> the same day, rescaled by actions that had not happened yet

Run:  uv run python demo/vintage_drift.py
Out:  a table on stdout + docs/media/vintage_drift.svg
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from stock_analyst_mcp import data as D

# Dates chosen to straddle a real corporate action, so the drift is visible:
# RELIANCE 1:1 bonus 2024-10-28, HDFCBANK 1:1 bonus 2025-08-26,
# TCS bonus 2018-05-31, INFY dividends only in the window.
CASES = [
    ("RELIANCE.NS", "2024-10-01"),
    ("HDFCBANK.NS", "2025-08-01"),
    ("TCS.NS", "2018-05-01"),
    ("INFY.NS", "2020-01-01"),
]

OUT = Path(__file__).resolve().parent.parent / "docs" / "media" / "vintage_drift.svg"


def measure(symbol: str, as_of: str) -> dict:
    """Return both answers for one (symbol, date)."""
    raw = D.fetch_history(symbol)          # raw OHLC + vendor AdjClose, cached
    actions = D.fetch_actions(symbol)      # dividends / splits, cached
    cutoff = pd.Timestamp(as_of).date()

    hist = D.enforce_as_of(raw, cutoff)    # rows <= cutoff, nothing after
    adj, applied = D.rebuild_adjusted_with_actions(raw, actions, cutoff)

    known_then = float(adj.loc[hist.index[-1]])
    adj_col = "Adj Close" if "Adj Close" in hist.columns else "AdjClose"
    vendor_today = float(hist[adj_col].iloc[-1])
    return {
        "symbol": symbol,
        "date": hist.index[-1].date().isoformat(),
        "known_then": round(known_then, 2),
        "vendor_today": round(vendor_today, 2),
        "drift_pct": round((vendor_today - known_then) / known_then * 100, 1),
        # actions the cutoff ALLOWED into the rebuild (all others are invisible)
        "applied_actions": len(applied),
    }


def svg(rows: list[dict]) -> str:
    """Dependency-free bar chart of |drift| per case (renders on GitHub)."""
    W, H, LEFT, TOP, BAR = 760, 300, 190, 46, 26
    max_abs = max(abs(r["drift_pct"]) for r in rows) or 1.0
    usable = W - LEFT - 90
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="Segoe UI, Helvetica, Arial">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{LEFT}" y="26" font-size="15" font-weight="600" fill="#111">'
        f'Same day, same stock: as known then vs as adjusted today</text>',
    ]
    for i, r in enumerate(rows):
        y = TOP + i * (BAR + 22)
        w = max(2.0, abs(r["drift_pct"]) / max_abs * usable)
        color = "#c0392b" if r["drift_pct"] < 0 else "#1f7a4d"
        label = f'{r["symbol"]}  {r["date"]}'
        parts += [
            f'<text x="12" y="{y + 18}" font-size="12.5" fill="#222">{label}</text>',
            f'<rect x="{LEFT}" y="{y}" width="{w:.1f}" height="{BAR}" rx="3" fill="{color}" opacity="0.85"/>',
            f'<text x="{LEFT + w + 8:.1f}" y="{y + 18}" font-size="12.5" '
            f'font-weight="600" fill="{color}">{r["drift_pct"]:+.1f}%</text>',
        ]
    parts += [
        f'<text x="12" y="{TOP + len(rows) * (BAR + 22) + 16}" font-size="11.5" fill="#666">'
        f'red = today\u2019s adjusted price sits below the close printed that day (later dividends applied retroactively)</text>',
        "</svg>",
    ]
    return "\n".join(parts)


def main() -> int:
    rows = [measure(s, d) for s, d in CASES]

    print(f'{"symbol":14s} {"as_of":12s} {"known_then":>12s} {"vendor_today":>13s} {"drift":>8s}')
    for r in rows:
        print(f'{r["symbol"]:14s} {r["date"]:12s} {r["known_then"]:12.2f} '
              f'{r["vendor_today"]:13.2f} {r["drift_pct"]:+7.1f}%')

    worst = max(rows, key=lambda r: abs(r["drift_pct"]))
    print(
        f'\n{worst["symbol"]} on {worst["date"]}: the close printed that day was '
        f'{worst["known_then"]:.2f} INR; today\u2019s adjusted series reports '
        f'{worst["vendor_today"]:.2f} INR for the same day ({worst["drift_pct"]:+.1f}%).'
    )
    print("A threshold, a stop level, or a position size computed on the second")
    print("number is a decision made with information from the future.")
    print()
    print("Note on splits: yfinance's Close is already split-adjusted, so this")
    print("server does NOT re-apply split ratios (doing so halves pre-split rows).")
    print("The drift measured here is dividend vintage. A pre-split price as filed")
    print("cannot be recovered from this source at all \u2014 the server records each")
    print("split in adjustment_history with applied: false instead of inventing one.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(svg(rows), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(OUT.parent.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
