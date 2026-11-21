"""indian-stocks-mcp — MCP server for India-only (NSE/BSE) stock analysis.

Stdio transport. Never ``print()`` to stdout (corrupts JSON-RPC). All
logging goes to stderr. Every tool output carries ``source``, ``as_of``
(the effective cutoff, echoed), and ``currency``.

Tools (all take ``as_of`` as an optional YYYY-MM-DD cutoff):
  get_ohlcv, get_returns, get_technical_indicators, get_chart_patterns,
  get_risk_metrics, get_index_history, get_peer_comparison
"""

from __future__ import annotations

import logging
import sys
from importlib import metadata
from typing import Any

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

from .data import (
    SOURCE,
    WATCHLIST,
    FutureDateError,
    UnknownSymbolError,
    _get_frame,
    log_to_stderr,
    normalise_symbol,
)
from .technicals import compute_all

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("stock_analyst_mcp.server")

try:
    _PACKAGE_VERSION = metadata.version("indian-stocks-mcp")
except metadata.PackageNotFoundError:  # not installed (e.g. bare src checkout)
    _PACKAGE_VERSION = "0.3.0"  # literal fallback: keep in sync with pyproject.toml

mcp = FastMCP(
    "indian-stocks-mcp",  # product identity (import package stays stock_analyst_mcp)
    instructions=(
        "India-only (NSE/BSE) stock analysis over historical price data. "
        "Prefer get_ohlcv/get_returns/get_technical_indicators for price questions. "
        "Use get_chart_patterns for 'any hammer/doji/support/resistance?' questions. "
        "Pass as_of (YYYY-MM-DD) to ask 'as of that date' — results are cut off "
        "at that date and adjusted prices rebuilt only from actions known by then. "
        "Do not invent numbers; compute from tool output."
    ),
)

# FastMCP 1.12.4 has no ``version`` parameter and does not forward one to the
# lowlevel Server, so serverInfo.version would report the *SDK* version
# (pkg_version("mcp") -> "1.12.4"). The lowlevel Server exposes a plain
# ``version`` attribute that the initialize handler reads
# (mcp/server/lowlevel/server.py:145,174), so setting it here is the only
# route to make serverInfo.version reflect *this* package's version.
mcp._mcp_server.version = _PACKAGE_VERSION


def _meta(as_of_eff: str, data_through: str, extra: dict[str, Any] | None = None,
          rows_used: int | None = None) -> dict[str, Any]:
    """Provenance envelope for every tool.

    ``rows_used`` = rows available after as_of filtering, BEFORE the
    lookback tail-trim (see data._get_frame). It is the honest "how much
    data existed at the cutoff" number; the returned slice may be shorter.
    """
    m: dict[str, Any] = {
        "source": SOURCE,
        "as_of": as_of_eff,
        "data_through": data_through,
        "currency": "INR",
        "rows_used": rows_used,
    }
    if extra:
        m.update(extra)
    return m


def _err(message: str) -> dict[str, Any]:
    return {"isError": True, "error": message, "source": SOURCE}


@mcp.tool()
def get_ohlcv(symbol: str, as_of: str | None = None, lookback_days: int = 90) -> dict[str, Any]:
    """Daily OHLC (adjusted close rebuilt as-of) for an NSE/BSE symbol."""
    try:
        df, d, history, _ = _get_frame(symbol, as_of, lookback_days or 90)
        as_of_eff = str(d) if d else str(df.index.max().date())
        rows = [
            {
                "date": str(idx.date()),
                "open": round(float(r["Open"]), 4),
                "high": round(float(r["High"]), 4),
                "low": round(float(r["Low"]), 4),
                "close": round(float(r["Close"]), 4),
                "close_adj": round(float(r["CloseAdj"]), 4),
                "volume": int(r["Volume"]),
            }
            for idx, r in df.iterrows()
        ]
        return {
            "symbol": normalise_symbol(symbol),
            "rows": rows,
            "n_rows": len(rows),
            "adjustment_history": history,
            **_meta(as_of_eff, str(df.index.max().date()), rows_used=df.attrs.get("rows_used")),
        }
    except (UnknownSymbolError, FutureDateError, ValueError) as exc:
        return _err(str(exc))


@mcp.tool()
def get_returns(symbol: str, as_of: str | None = None, windows: list[int] | None = None) -> dict[str, Any]:
    """Simple/log returns over lookback windows, computed in pandas."""
    wins = windows if windows else [1, 5, 20, 60]
    try:
        df, d, _, _ = _get_frame(symbol, as_of, max(wins) + 10)
        closes = df["CloseAdj"].astype(float)
        last = closes.iloc[-1]
        results = {}
        for w in wins:
            if len(closes) <= w:
                results[f"window_{w}d"] = None
                continue
            past = closes.iloc[-1 - w]
            simple = (last - past) / past if past != 0 else None
            logr = float(np.log(last / past)) if past > 0 else None
            results[f"window_{w}d"] = {
                "simple": round(float(simple), 6) if simple is not None else None,
                "log": round(logr, 6) if logr is not None else None,
                "pct_simple": round(float(simple) * 100, 4) if simple is not None else None,
            }
        return {
            "symbol": normalise_symbol(symbol),
            "returns": results,
            **_meta(str(d) if d else str(closes.index[-1].date()), str(closes.index[-1].date()), rows_used=df.attrs.get("rows_used")),
        }
    except (UnknownSymbolError, FutureDateError, ValueError) as exc:
        return _err(str(exc))


@mcp.tool()
def get_technical_indicators(symbol: str, as_of: str | None = None,
                             indicators: list[str] | None = None) -> dict[str, Any]:
    """RSI/SMA/EMA/MACD/Bollinger/ATR/VWAP computed in Python, never LLM math."""
    inds = indicators or ["SMA_20", "SMA_50", "EMA_12", "EMA_26", "RSI_14", "MACD", "BOLLINGER", "ATR_14", "VWAP"]
    try:
        df, d, _, _ = _get_frame(symbol, as_of, 400)
        calc = compute_all(df, inds)
        last_close = float(df["CloseAdj"].iloc[-1])
        sma50 = calc.get("SMA_50")
        sma200 = calc.get("SMA_200")
        trend = "bullish" if (sma50 is not None and sma200 is not None and sma50 > sma200) else "bearish"
        out = {k: (round(float(v), 4) if v is not None and np.isfinite(float(v)) else None)
               for k, v in calc.items() if v is not None or k in ("RSI_14", "MACD_SIGNAL")}
        return {
            "symbol": normalise_symbol(symbol),
            "current_price": round(last_close, 4),
            "trend": trend,
            "indicators": out,
            **_meta(str(d) if d else str(df.index[-1].date()), str(df.index[-1].date()),
                    extra={"computed_in": "pandas/numpy (no LLM math)"},
                    rows_used=df.attrs.get("rows_used")),
        }
    except (UnknownSymbolError, FutureDateError, ValueError) as exc:
        return _err(str(exc))


@mcp.tool()
def get_chart_patterns(symbol: str, as_of: str | None = None, lookback_days: int = 120,
                       window: int = 20) -> dict[str, Any]:
    """Candlestick patterns (hammer, doji, engulfing, stars) + support/resistance.

    ``window`` = trailing-bar scan width for the candlestick detector
    (default 20). A pattern older than the last ``window`` bars cannot be
    reported — raise ``window`` to look further back (bounded by lookback_days).
    """
    try:
        from .technicals import compute_chart_patterns

        if window is None or int(window) < 1:
            return _err("window must be a positive integer")
        df, d, _, _ = _get_frame(symbol, as_of, lookback_days or 120)
        patterns = compute_chart_patterns(df, window=int(window))
        return {
            "symbol": normalise_symbol(symbol),
            "patterns": patterns["patterns"],
            "support_resistance": patterns["support_resistance"],
            "scan_window": int(window),
            **_meta(str(d) if d else str(df.index[-1].date()), str(df.index[-1].date()),
                    rows_used=df.attrs.get("rows_used")),
        }
    except (UnknownSymbolError, FutureDateError, ValueError) as exc:
        return _err(str(exc))


@mcp.tool()
def get_risk_metrics(symbol: str, as_of: str | None = None, benchmark: str = "^NSEI",
                     lookback_days: int = 252) -> dict[str, Any]:
    """Sharpe, Sortino, max drawdown, volatility vs benchmark."""
    try:
        df, d, _, _ = _get_frame(symbol, as_of, lookback_days or 252)
        sym_ret = df["CloseAdj"].astype(float).pct_change().dropna()
        bench = None
        try:
            bdf, _, _, _ = _get_frame(benchmark, as_of, lookback_days or 252)
            bench_ret = bdf["CloseAdj"].astype(float).pct_change().dropna()
            bench = bench_ret
        except Exception:
            pass
        ann = 252
        vol = float(sym_ret.std() * np.sqrt(ann)) if len(sym_ret) > 1 else None
        mean = float(sym_ret.mean() * ann) if len(sym_ret) else None
        sharpe = (mean / vol) if (vol and vol > 0) else None
        downside = sym_ret[sym_ret < 0].std()
        sortino = (mean / (float(downside) * np.sqrt(ann))) if (downside and float(downside) > 0) else None
        cum = (1 + sym_ret).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        min_idx = dd.idxmin()
        max_dd = float(dd.min()) if len(dd) else None
        dd_end = str(min_idx.date()) if len(dd) else None
        dd_start = str(cum.loc[:min_idx].idxmax().date()) if len(dd) and dd_end else None
        beta = None
        alpha = None
        if bench is not None and len(bench) > 1:
            aligned = pd.concat([sym_ret, bench], axis=1, join="inner").dropna()
            if len(aligned) > 1:
                cov = float(aligned.cov().iloc[0, 1])
                var = float(aligned.iloc[:, 1].var())
                beta = cov / var if var > 0 else None
                if beta is not None and mean is not None:
                    alpha = mean - beta * float(bench.mean() * ann)
        return {
            "symbol": normalise_symbol(symbol),
            "benchmark": benchmark,
            "sharpe": round(sharpe, 4) if sharpe is not None else None,
            "sortino": round(sortino, 4) if sortino is not None else None,
            "max_drawdown": round(max_dd, 4) if max_dd is not None else None,
            "drawdown_start": dd_start,
            "drawdown_end": dd_end,
            "volatility_annualised": round(vol, 4) if vol is not None else None,
            "mean_annual_return": round(mean, 4) if mean is not None else None,
            "beta": round(beta, 4) if beta is not None else None,
            "alpha_vs_benchmark": round(alpha, 4) if alpha is not None else None,
            "n_observations": int(len(sym_ret)),
            **_meta(str(d) if d else str(sym_ret.index[-1].date()) if len(sym_ret) else "",
                    str(df.index[-1].date()), rows_used=df.attrs.get("rows_used")),
        }
    except (UnknownSymbolError, FutureDateError, ValueError) as exc:
        return _err(str(exc))


@mcp.tool()
def get_index_history(index: str, as_of: str | None = None, lookback_days: int = 90) -> dict[str, Any]:
    """Index close history (^NSEI, ^BSESN, ^NSEBANK …)."""
    try:
        df, d, _, _ = _get_frame(index, as_of, lookback_days or 90)
        rows = [
            {"date": str(idx.date()), "close": round(float(r["CloseAdj"]), 4),
             "volume": int(r["Volume"]) if pd.notna(r["Volume"]) else None}
            for idx, r in df.iterrows()
        ]
        return {
            "index": normalise_symbol(index),
            "rows": rows,
            "n_rows": len(rows),
            **_meta(str(d) if d else str(df.index[-1].date()), str(df.index[-1].date()),
                    rows_used=df.attrs.get("rows_used")),
        }
    except (UnknownSymbolError, FutureDateError, ValueError) as exc:
        return _err(str(exc))


@mcp.tool()
def get_peer_comparison(symbols: list[str], as_of: str | None = None,
                        metric: str = "return_20d") -> dict[str, Any]:
    """Side-by-side comparison of NSE/BSE peers on one metric."""
    if not symbols or not isinstance(symbols, (list, tuple)):
        return _err("symbols must be a non-empty list")
    if metric not in ("return_20d", "return_60d", "vol_60d", "sharpe_60d", "drawdown_60d", "rsi_14"):
        return _err(f"Unsupported metric '{metric}'. Use return_20d, return_60d, vol_60d, sharpe_60d, drawdown_60d, rsi_14")
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    as_of_eff = as_of or ""
    df = None  # last successful frame, for data_through
    for raw in symbols:
        try:
            df, d, _, _ = _get_frame(raw, as_of, 120)
            closes = df["CloseAdj"].astype(float)
            value: float | None = None
            if metric in ("return_20d", "return_60d"):
                w = 20 if metric == "return_20d" else 60
                if len(closes) > w and closes.iloc[-1 - w] != 0:
                    value = (closes.iloc[-1] - closes.iloc[-1 - w]) / closes.iloc[-1 - w]
            elif metric == "vol_60d":
                rets = closes.pct_change().dropna().tail(60)
                value = float(rets.std() * np.sqrt(252)) if len(rets) > 1 else None
            elif metric == "sharpe_60d":
                rets = closes.pct_change().dropna().tail(60)
                if len(rets) > 1:
                    ann = float(rets.mean() * 252)
                    vol = float(rets.std() * np.sqrt(252))
                    value = ann / vol if vol > 0 else None
            elif metric == "drawdown_60d":
                rets = closes.pct_change().dropna().tail(60)
                cum = (1 + rets).cumprod()
                value = float(((cum - cum.cummax()) / cum.cummax()).min()) if len(cum) else None
            elif metric == "rsi_14":
                calc = compute_all(df, ["RSI_14"])
                v = calc.get("RSI_14")
                value = float(v) if v is not None and np.isfinite(float(v)) else None
            results.append({
                "symbol": normalise_symbol(raw),
                "value": round(value, 6) if value is not None else None,
                "current_price": round(float(closes.iloc[-1]), 4),
                "currency": "INR",
            })
            if d is not None:
                as_of_eff = str(d)
        except (UnknownSymbolError, FutureDateError, ValueError) as exc:
            errors.append({"symbol": raw, "error": str(exc)})
    # ranks (descending value; None last)
    ranked = sorted(results, key=lambda r: (r["value"] is None, -r["value"] if r["value"] is not None else 0))
    for i, r in enumerate(ranked, start=1):
        r["rank"] = i
    return {
        "metric": metric,
        "results": results,
        "ranked": ranked,
        "errors": errors,
        **_meta(as_of_eff or "latest", str(df.index[-1].date()) if df is not None and not df.empty else "",
               rows_used=df.attrs.get("rows_used") if df is not None else None),
    }


@mcp.resource("stock://symbols/watchlist")
def watchlist_resource() -> str:
    """Watchlist of NSE symbols in scope."""
    return "\n".join(WATCHLIST)


@mcp.resource("stock://{symbol}/profile")
def profile_resource(symbol: str) -> str:
    """Text profile for a symbol (exchange, currency, market hours).

    Honesty note: the exchange is derived from the ticker suffix only
    (".NS" -> NSE, ".BO" -> BSE) and is NOT verified against the exchange.
    The ``Mapping`` line says so; agents/dashboards should display it
    alongside the exchange rather than presenting it as verified fact.
    """
    sym = normalise_symbol(symbol)
    exchange = "NSE" if sym.endswith(".NS") else "BSE" if sym.endswith(".BO") else "INDEX"
    return (
        f"Symbol: {sym}\n"
        f"Exchange: {exchange}\n"
        f"Mapping: suffix-based, not verified against the exchange\n"
        f"Currency: INR\n"
        f"Market hours: 09:15-15:30 IST (Mon-Fri, NSE/BSE)\n"
        f"Source: {SOURCE}\n"
    )


def main() -> None:
    log_to_stderr("indian-stocks-mcp starting (stdio) …")
    mcp.run()


if __name__ == "__main__":
    main()
