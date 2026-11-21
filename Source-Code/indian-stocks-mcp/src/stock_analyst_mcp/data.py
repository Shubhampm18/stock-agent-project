"""Data layer for indian-stocks-mcp.

yfinance fetching, local Parquet caching (24h TTL), point-in-time enforcement,
and the adjustment-vintage rebuild (adjusted close recomputed from corporate
actions known as of the cutoff, so a later split cannot change an earlier
price). Nothing here returns data after the caller's cutoff.

Symbols: bare ticker (``RELIANCE``) is an NSE guess — ``.NS`` is appended.
Explicit ``.BO`` is required for BSE. Non-Indian/unknown bare tickers will
still attempt ``.NS`` and then error with "Did you mean...?" suggestions.
"""

from __future__ import annotations

import difflib
import logging
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger("stock_analyst_mcp.data")
log.addHandler(logging.StreamHandler(sys.stderr))

WATCHLIST = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "ITC.NS",
    "TATAMOTORS.NS",
]

CACHE_DIR = Path.home() / ".cache" / "stock-analyst-mcp"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SOURCE = "yfinance"


class UnknownSymbolError(ValueError):
    """Raised when a symbol is not found."""


class FutureDateError(ValueError):
    """Raised when as_of is beyond the available data."""


def log_to_stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def _parse_as_of(as_of: str | None) -> date | None:
    """Parse YYYY-MM-DD; None stays None. Raises FutureDateError on future.

    The future check uses Asia/Kolkata today, not the host-local date, so
    an as_of that is "tomorrow in IST" is correctly rejected even when the
    host clock is a day behind.
    """
    if as_of is None:
        return None
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", as_of):
        raise ValueError("as_of must be YYYY-MM-DD")
    try:
        d = date.fromisoformat(as_of)
    except ValueError:
        raise ValueError("as_of must be YYYY-MM-DD")
    ist_today = pd.Timestamp.now(tz="Asia/Kolkata").date()
    if d > ist_today:
        raise FutureDateError(f"as_of {as_of} is in the future")
    return d


def normalise_symbol(symbol: str) -> str:
    """Accept RELIANCE / RELIANCE.NS / RELIANCE.BO / RELIANCE.bo etc.

    Bare ticker (``RELIANCE``) is an NSE guess — ``.NS`` is appended.
    Explicit ``.BO`` is required for BSE. Non-Indian/unknown bare tickers
    will still attempt ``.NS`` and then error with "Did you mean...?".
    """
    s = symbol.strip().upper()
    if s.endswith(".NS") or s.endswith(".BO"):
        return s
    if "." in s:  # e.g. RELIANCE. (typo) — treat as NSE default
        s = s.split(".")[0]
    # bare ticker without exchange: default to NSE India (.NS)
    if s in {"TCS", "INFY", "RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN", "ITC",
             "TATAMOTORS", "TATAMOTORS.NS"}:
        return f"{s}.NS" if not s.endswith(".NS") else s
    # index aliases (alias table kept in sync with memory._default_normalise_symbol
    # in stock-analyst-app — see the parity test there)
    alias = {"NIFTY": "^NSEI", "NIFTY50": "^NSEI", "NIFTY_50": "^NSEI",
             "NIFTY 50": "^NSEI", "BANKNIFTY": "^NSEBANK",
             "SENSEX": "^BSESN", "NIFTY BANK": "^NSEBANK"}
    if s in alias:
        return alias[s]
    if s.startswith("^"):  # keep index tickers as-is
        return s
    return f"{s}.NS"


def _did_you_mean(symbol: str) -> list[str]:
    """Prefix/substring matches merged with fuzzy matches (difflib), deduped.

    difflib catches transpositions ("RELIIANCE" -> "RELIANCE.NS") that
    prefix matching alone misses.
    """
    s = re.sub(r"\.(NS|BO)$", "", symbol.strip().upper())
    if not s:
        return []
    prefix = [w for w in WATCHLIST if w.startswith(s) or s in w]
    fuzzy = difflib.get_close_matches(s, [re.sub(r"\.(NS|BO)$", "", w) for w in WATCHLIST],
                                      n=3, cutoff=0.6)
    fuzzy_full = [f"{t}.NS" if f"{t}.NS" in WATCHLIST else t for t in fuzzy]
    seen: set[str] = set()
    out: list[str] = []
    for w in prefix + fuzzy_full:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out[:3]


def _cache_path(symbol: str) -> Path:
    name = re.sub(r"[^A-Z0-9^.-]", "_", symbol.upper())
    return CACHE_DIR / f"{name}.parquet"


def fetch_history(symbol: str, period: str = "max", not_found_ok: bool = False,
                  force_refresh: bool = False, ttl_hours: float = 24) -> pd.DataFrame:
    """Fetch daily OHLC (unadjusted) from yfinance, cached to Parquet (24h TTL).

    Returns DataFrame indexed by date with columns: Open, High, Low, Close,
    Volume, AdjClose, Dividends, Stock Splits.
    Raises UnknownSymbolError on empty result.
    """
    sym = normalise_symbol(symbol)
    cache = _cache_path(sym)
    bare_assumed = symbol.strip().upper() not in (sym, sym.replace(".NS", ""), sym.replace(".BO", ""))
    if cache.exists() and period == "max" and not force_refresh:
        age_h = (pd.Timestamp.now(tz="Asia/Kolkata") - pd.Timestamp.fromtimestamp(cache.stat().st_mtime, tz="Asia/Kolkata")).total_seconds() / 3600
        if age_h < ttl_hours:
            try:
                df = pd.read_parquet(cache)
                if not df.empty:
                    return _drop_unfinalised_rows(df)
            except Exception as exc:
                log.warning("cache read failed for %s: %s", sym, exc)
        else:
            log.warning("cache stale (%.1fh > %.1fh TTL) for %s — refetching", age_h, ttl_hours, sym)

    try:
        hist = yf.Ticker(sym).history(period=period, auto_adjust=False)
    except Exception as exc:  # network etc.
        if cache.exists() and period == "max":
            log.warning("refetch failed for %s (%s) — falling back to stale cache", sym, exc)
            try:
                df = pd.read_parquet(cache)
                if not df.empty:
                    return df
            except Exception as cache_exc:
                log.warning("stale cache also unreadable for %s: %s", sym, cache_exc)
        raise UnknownSymbolError(f"Could not fetch data for {symbol}: {exc}") from exc
    if hist is None or hist.empty:
        if not_found_ok:
            return hist  # empty
        suggestions = _did_you_mean(symbol)
        hint = f" Did you mean {', '.join(suggestions)}?" if suggestions else ""
        if bare_assumed:
            hint += " Note: bare tickers are assumed to be NSE (.NS); for BSE use the .BO suffix explicitly."
        raise UnknownSymbolError(f"Unknown symbol: {symbol}.{hint}")

    # Normalise index
    df = hist.copy()
    df.index = pd.to_datetime(df.index).tz_convert("Asia/Kolkata").tz_localize(None)
    df = df.sort_index()
    df = _drop_unfinalised_rows(df)
    if period == "max" and cache.parent.exists():
        try:
            df.to_parquet(cache)
        except Exception as exc:
            log.warning("cache write failed for %s: %s", sym, exc)
    return df


def _drop_unfinalised_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NaN/zero Close (yfinance sometimes appends an
    unfinalised current-day row whose OHLC is empty). One such row poisons
    every downstream consumer: last close, returns, and the S/R pivot
    comparisons (``p < NaN`` is always False, so all levels vanish).
    Volume-only NaN is tolerated (indices carry no volume)."""
    if df.empty or "Close" not in df.columns:
        return df
    close = pd.to_numeric(df["Close"], errors="coerce")
    keep = close.notna() & (close != 0)
    dropped = int((~keep).sum())
    if dropped:
        log.warning("dropped %d unfinalised row(s) with empty Close", dropped)
    return df.loc[keep]


def fetch_actions(symbol: str) -> pd.DataFrame:
    """Dividend/split actions with ex-dates, for adjustment-history output."""
    sym = normalise_symbol(symbol)
    try:
        actions = yf.Ticker(sym).actions
    except Exception:
        return pd.DataFrame(columns=["Dividends", "Stock Splits"])
    if actions is None or actions.empty:
        return pd.DataFrame(columns=["Dividends", "Stock Splits"])
    actions = actions.copy()
    actions.index = pd.to_datetime(actions.index).tz_convert("Asia/Kolkata").tz_localize(None)
    return actions.sort_index()


def rebuild_adjusted_with_actions(df: pd.DataFrame, actions: pd.DataFrame,
                                  as_of: date | None) -> tuple[pd.Series, list[dict[str, Any]]]:
    """Rebuild adjusted close on a *split-adjusted* price series.

    Vendor convention (verified against yfinance 1.8): ``Close`` is **already
    split-adjusted** and **not** dividend-adjusted; ``Adj Close`` is both.
    Therefore split ratios must NOT be re-applied to ``Close`` — doing so
    halves every pre-split row and manufactures a fake +100% jump on the
    ex-date (the bug this function used to have). Splits are recorded in
    ``adjustment_history`` as ``applied: False`` with the reason.

    Dividends are as-filed (pre-split basis), so each one is rebased onto the
    current share basis before the standard factor method is applied:

        effective = amount / product(split_ratio for splits with ex_date > ex)
        factor    = 1 - effective / close_on_day_before_ex

    All pre-ex prices are multiplied by the cumulative factor. For dividends,
    actions with ex_date > as_of are skipped, so a later dividend cannot change
    an earlier price — the point-in-time (vintage) guarantee.

    Limitation, stated rather than hidden: because ``Close`` is retroactively
    split-adjusted upstream, the as-filed *price* of a pre-split date is not
    recoverable from this source. Ratios and returns are unaffected; absolute
    pre-split levels are in the current share basis. ``adjustment_history``
    reports this per split.
    """
    raw = df["Close"].astype(float).copy()
    adj = raw.copy()
    history: list[dict[str, Any]] = []
    if actions is None or actions.empty:
        return adj, history

    # Split ratios by ex-date: needed only to rebase as-filed dividends.
    split_ratios: dict[date, float] = {}
    for ex_date, row in actions.iterrows():
        ratio = float(row.get("Stock Splits", 0.0)) if "Stock Splits" in row else 0.0
        if ratio and ratio > 0:
            split_ratios[pd.Timestamp(ex_date).date()] = ratio

    for ex_date, row in actions.iterrows():
        ex = pd.Timestamp(ex_date).date()
        if as_of is not None and ex > as_of:
            continue  # later action cannot affect earlier adjustment

        split = split_ratios.get(ex, 0.0)
        if split:
            history.append({
                "ex_date": str(ex), "type": "split", "ratio": split,
                "applied": False,
                "reason": "vendor Close is already split-adjusted; "
                          "as-filed pre-split price not recoverable from source",
            })

        div = float(row.get("Dividends", 0.0)) if "Dividends" in row else 0.0
        if div and div > 0:
            # Rebase the as-filed cash into the current (post-split) share basis.
            scale = 1.0
            for split_ex, ratio in split_ratios.items():
                if split_ex > ex:
                    scale *= ratio
            effective = div / scale

            pre = raw[raw.index.date < ex]
            if pre.empty:
                continue
            close_before = float(pre.iloc[-1])
            if close_before <= 0:
                continue
            factor = 1.0 - effective / close_before
            history.append({"ex_date": str(ex), "type": "dividend",
                            "amount": round(div, 4),
                            "effective_amount": round(effective, 4),
                            "factor": round(factor, 8)})
            mask = adj.index.date < ex
            adj.loc[mask] = adj.loc[mask] * factor
    return adj, history


def enforce_as_of(df: pd.DataFrame, as_of: date | None) -> pd.DataFrame:
    """Hard-filter df to rows with date <= as_of; assert the cutoff held."""
    if as_of is None:
        return df
    cutoff = pd.Timestamp(as_of)
    filtered = df[df.index <= cutoff]
    if filtered.empty:
        raise FutureDateError(f"as_of {as_of} is before all available data")
    if not (filtered.index.max() <= cutoff):
        raise AssertionError("enforce_as_of failed: future row leaked")
    return filtered


def _get_frame(symbol: str, as_of: str | None, lookback_days: int = 90):
    """Fetch, filter to as_of, and take the final lookback_days rows.

    Returns ``(filtered, d, history, actions)`` — the tuple arity is part of
    the contract consumed by server tools, the dashboard, and tests.

    The returned frame carries ``attrs["rows_used"]``: the number of rows
    available after as_of filtering **before** the lookback tail-trim. That
    is the honest provenance number — it tells the caller how much data
    existed at the cutoff, even when the returned slice is shorter (a
    lookback_days larger than the available history yields
    rows_used == len(returned frame); a smaller lookback yields
    rows_used > len(returned frame)).
    """
    if lookback_days is None or lookback_days <= 0:
        lookback_days = 90
    df = fetch_history(symbol)
    actions = fetch_actions(symbol)
    d = _parse_as_of(as_of)
    filtered = enforce_as_of(df, d)
    rows_used = len(filtered)  # pre-trim: everything the cutoff admits
    adj_close, history = rebuild_adjusted_with_actions(df, actions, d)
    # Attach rebuilt adjusted close (aligned to filtered rows)
    filtered = filtered.copy()
    filtered["CloseAdj"] = adj_close.loc[filtered.index]
    # Last lookback_days rows
    filtered = filtered.tail(lookback_days)
    # Set AFTER tail() so propagation quirks can't drop it.
    filtered.attrs["rows_used"] = rows_used
    return filtered, d, history, actions


def _is_trading_day_note(close_series: pd.Series, as_of: date | None) -> str:
    if close_series.empty:
        return ""
    last = close_series.index.max().date()
    if as_of is not None and last < as_of:
        # last trading day before as_of (weekend/holiday roll-back)
        return f"Last trading day on/before {as_of}: {last}"
    return ""


__all__ = [
    "WATCHLIST",
    "SOURCE",
    "UnknownSymbolError",
    "FutureDateError",
    "normalise_symbol",
    "fetch_history",
    "fetch_actions",
    "enforce_as_of",
    "rebuild_adjusted_with_actions",
    "_get_frame",
    "log_to_stderr",
]
