"""Technical indicators in Python (pandas/numpy) — never LLM math."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> float | None:
    if len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    if avg_loss.iloc[-1] == 0:
        return 100.0
    rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
    return float(100 - 100 / (1 + rs))


def _macd(close: pd.Series) -> tuple[float | None, float | None, float | None]:
    if len(close) < 35:
        return None, None, None
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal
    return float(macd_line.iloc[-1]), float(signal.iloc[-1]), float(histogram.iloc[-1])


def compute_all(df: pd.DataFrame, indicators: list[str]) -> dict[str, float | None]:
    """Compute the requested indicators; missing data yields None, never NaN."""
    close = df["CloseAdj"].astype(float) if "CloseAdj" in df else df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    vol = df["Volume"].astype(float)
    out: dict[str, float | None] = {}

    for ind in indicators:
        key = ind.upper().replace("-", "_")
        if key == "SMA_20" and len(close) >= 20:
            out[key] = float(close.rolling(20).mean().iloc[-1])
        elif key == "SMA_50" and len(close) >= 50:
            out[key] = float(close.rolling(50).mean().iloc[-1])
        elif key == "SMA_200" and len(close) >= 200:
            out[key] = float(close.rolling(200).mean().iloc[-1])
        elif key == "EMA_12" and len(close) >= 12:
            out[key] = float(close.ewm(span=12, adjust=False).mean().iloc[-1])
        elif key == "EMA_26" and len(close) >= 26:
            out[key] = float(close.ewm(span=26, adjust=False).mean().iloc[-1])
        elif key == "RSI_14":
            out[key] = _rsi(close, 14)
        elif key == "MACD":
            m, s, h = _macd(close)
            out[key] = m
            out["MACD_SIGNAL"] = s
            out["MACD_HISTOGRAM"] = h
        elif key == "BOLLINGER" and len(close) >= 20:
            mid = float(close.rolling(20).mean().iloc[-1])
            std = float(close.rolling(20).std().iloc[-1])
            if np.isfinite(mid) and np.isfinite(std):
                out["BOLLINGER_MID"] = mid
                out["BOLLINGER_UPPER"] = mid + 2 * std
                out["BOLLINGER_LOWER"] = mid - 2 * std
        elif key == "ATR_14" and len(close) >= 15:
            prev = close.shift(1)
            tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            out[key] = float(atr) if np.isfinite(atr) else None
        elif key == "VWAP" and len(close) >= 1 and vol.sum() > 0:
            tp = (high + low + close) / 3
            out[key] = float((tp * vol).sum() / vol.sum()) if vol.sum() > 0 else None
        else:
            out.setdefault(key, None)

    # Always include SMA_200 when enough data for the trend field
    if "SMA_200" not in out and len(close) >= 200:
        out["SMA_200"] = float(close.rolling(200).mean().iloc[-1])
    return out


# ---------------------------------------------------------------------------
# Chart pattern detection (Stage 1: candlesticks + support/resistance)
# ---------------------------------------------------------------------------

def _body(o: float, c: float) -> float:
    return abs(c - o)


def _range_(h: float, l: float) -> float:
    return h - l


def _candlesticks(df: pd.DataFrame, window: int = 20) -> list[dict[str, object]]:
    """Scan the last ``window`` bars for single/two/three-bar candlestick
    predicates.

    Each predicate is an explicit threshold check on Open/High/Low/Close
    ratios; `confidence` is a fixed, documented strength for the shape
    (no fitted parameters, so detection is deterministic):

    - hammer: long lower wick (>= 2x body), tiny upper wick (<= 10% of
      range), close in the upper half of the range.
    - shooting star: mirror of hammer (long upper wick, tiny lower wick,
      close in the lower half).
    - doji: body <= 10% of the bar range (open ≈ close).
    - bullish engulfing: prior bar body fully contains the earlier bar's
      body and the prior bar closes up while the earlier closed down.
    - bearish engulfing: mirror (prior bar closes down, prior body
      contains the earlier up-bar's body).
    - morning star: 3-bar; middle bar's high gaps below the prior bar's
      low (star body entirely below), third bar closes back above the
      midpoint of the first bar's body.
    - evening star: mirror of morning star.
    """
    open_ = df["Open"].astype(float).to_numpy()
    high = df["High"].astype(float).to_numpy()
    low = df["Low"].astype(float).to_numpy()
    close = df["Close"].astype(float).to_numpy()
    dates = [str(d.date()) if hasattr(d, "date") else str(d)[:10] for d in df.index]
    out: list[dict[str, object]] = []
    n = len(df)
    start = max(1, n - max(1, int(window)))
    for i in range(start, n):
        o, h, l, c = open_[i], high[i], low[i], close[i]
        rng = _range_(h, l)
        if rng <= 0:
            continue
        body = _body(o, c)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        # doji: body is a negligible fraction of the range.
        if body <= 0.10 * rng:
            out.append({"name": "doji", "date": dates[i], "confidence": 0.7})
        # hammer: long lower shadow, short upper shadow, close near high.
        if lower_wick >= 2.0 * body and upper_wick <= 0.10 * rng and c >= (h + l) / 2:
            out.append({"name": "hammer", "date": dates[i], "confidence": 0.8})
        # shooting star: long upper shadow, short lower shadow, close near low.
        if upper_wick >= 2.0 * body and lower_wick <= 0.10 * rng and c <= (h + l) / 2:
            out.append({"name": "shooting_star", "date": dates[i], "confidence": 0.8})
        # engulfing: need a valid prior bar.
        if i >= 1:
            po, ph, pl, pc = open_[i - 1], high[i - 1], low[i - 1], close[i - 1]
            pbody = _body(po, pc)
            if pc < po and c > o and min(o, c) <= min(po, pc) and max(o, c) >= max(po, pc) and body > pbody:
                out.append({"name": "bullish_engulfing", "date": dates[i], "confidence": 0.85})
            elif pc > po and c < o and min(o, c) <= min(po, pc) and max(o, c) >= max(po, pc) and body > pbody:
                out.append({"name": "bearish_engulfing", "date": dates[i], "confidence": 0.85})
        # stars: need 3 bars (i-2, i-1, i).
        if i >= 2:
            o1, c1 = open_[i - 2], close[i - 2]
            h2, l2 = high[i - 1], low[i - 1]
            o2, c2 = open_[i - 1], close[i - 1]
            first_body_mid = (o1 + c1) / 2
            if c1 < o1 and h2 < low[i - 2] and c >= first_body_mid:
                out.append({"name": "morning_star", "date": dates[i], "confidence": 0.75})
            elif c1 > o1 and l2 > high[i - 2] and c <= first_body_mid:
                out.append({"name": "evening_star", "date": dates[i], "confidence": 0.75})
    return out


def _support_resistance(df: pd.DataFrame) -> dict[str, object]:
    """Pivot-based support/resistance from rolling-window extrema (window~14).

    A pivot high/low is a bar whose High/Low is the max/min over a centred
    window of 14 bars. Levels below the last close are supports, levels
    above are resistances; the nearest on each side is reported. Returns
    empty levels / None nearest on insufficient bars — never errors.
    """
    empty: dict[str, object] = {
        "support": [], "resistance": [],
        "nearest_support": None, "nearest_resistance": None,
    }
    if len(df) < 15:
        return empty
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    window = 14
    hi_pivots = high.rolling(window, center=True).max()
    lo_pivots = low.rolling(window, center=True).min()
    piv_highs = sorted({round(float(high[i]), 4) for i in high.index
                        if pd.notna(hi_pivots[i]) and float(high[i]) == float(hi_pivots[i])})
    piv_lows = sorted({round(float(low[i]), 4) for i in low.index
                       if pd.notna(lo_pivots[i]) and float(low[i]) == float(lo_pivots[i])})
    last_close = round(float(close.iloc[-1]), 4)
    support = [p for p in piv_lows if p < last_close]
    resistance = [p for p in piv_highs if p > last_close]
    return {
        "support": support,
        "resistance": resistance,
        "nearest_support": max(support) if support else None,
        "nearest_resistance": min(resistance) if resistance else None,
    }


def compute_chart_patterns(df: pd.DataFrame, window: int = 20) -> dict[str, object]:
    """Candlestick patterns + pivot support/resistance (pure, no I/O).

    ``window`` = how many trailing bars the candlestick scan inspects
    (default 20). A pattern older than the scan window is invisible by
    construction — pass a larger window to look further back; the caller
    is responsible for as_of truncation, and pattern dates are the input
    index dates, so they never exceed the frame's last date.

    Returns {"patterns": [{name, date, confidence}], "support_resistance":
    {support, resistance, nearest_support, nearest_resistance}}.
    """
    return {
        "patterns": _candlesticks(df, window=window),
        "support_resistance": _support_resistance(df),
    }


__all__ = ["compute_all", "compute_chart_patterns"]
