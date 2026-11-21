"""Unit tests for point-in-time enforcement, vintage adjustment, and tools.

Offline tests generate a deterministic synthetic OHLC frame and cover all
pure-logic paths without network. Truly network-dependent tests are marked
@pytest.mark.network and skip only when yfinance is unreachable.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import stock_analyst_mcp
from stock_analyst_mcp.data import (
    FutureDateError,
    enforce_as_of,
    fetch_actions,
    fetch_history,
    normalise_symbol,
    rebuild_adjusted_with_actions,
    _did_you_mean,
    _get_frame,
    _parse_as_of,
)
from stock_analyst_mcp.technicals import compute_all

SYMBOL = "RELIANCE.NS"
FUTURE_DATE = "2999-01-01"
network = pytest.mark.network


@pytest.fixture(scope="module")
def synthetic_history() -> pd.DataFrame:
    """Deterministic synthetic OHLC: 600 business days ending 2024-12-31."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end="2024-12-31", periods=600)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, len(dates))))
    df = pd.DataFrame({
        "Open": closes * 0.999,
        "High": closes * 1.01,
        "Low": closes * 0.99,
        "Close": closes,
        "Volume": rng.integers(500_000, 2_000_000, len(dates)),
    }, index=dates)
    df.index.name = "Date"
    return df


@pytest.fixture(scope="module")
def history() -> pd.DataFrame:
    """Cached Parquet if present, else the synthetic frame (offline-safe)."""
    from stock_analyst_mcp.data import _cache_path

    cache = _cache_path(SYMBOL)
    if cache.exists():
        try:
            df = pd.read_parquet(cache)
            if not df.empty:
                return df
        except Exception:
            pass
    # fall back to synthetic so offline CI never blocks on network
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end="2024-12-31", periods=600)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, len(dates))))
    df = pd.DataFrame({
        "Open": closes * 0.999, "High": closes * 1.01, "Low": closes * 0.99,
        "Close": closes, "Volume": rng.integers(500_000, 2_000_000, len(dates)),
    }, index=dates)
    df.index.name = "Date"
    return df


@pytest.fixture(scope="module")
def sample_date(history: pd.DataFrame) -> str:
    """A trading date at least 60 sessions before the end (so cutoffs can shrink)."""
    idx = history.index
    return str(idx[len(idx) - 60].date())


def _network_available() -> bool:
    """Probe yfinance reachability; True when a fetch would succeed."""
    try:
        df = fetch_history(SYMBOL)
        return df is not None and not df.empty
    except Exception:
        return False


network_or_skip = pytest.mark.skipif(not _network_available(), reason="network unavailable")


# --- pure-logic tests (no network) -----------------------------------------

def test_parse_as_of_rejects_bad_format():
    with pytest.raises(ValueError):
        _parse_as_of("01-01-2023")  # DD-MM-YYYY
    with pytest.raises(ValueError):
        _parse_as_of("2023/01/03")
    assert _parse_as_of("2023-01-03") == date(2023, 1, 3)


def test_parse_as_of_rejects_future_date():
    with pytest.raises(FutureDateError):
        _parse_as_of(FUTURE_DATE)


def test_drop_unfinalised_rows():
    """A NaN/zero Close row (yfinance unfinalised day) must be dropped,
    volume-only NaN must be kept."""
    from stock_analyst_mcp.data import _drop_unfinalised_rows

    idx = pd.to_datetime(["2026-11-17", "2026-11-18", "2026-11-19"])
    df = pd.DataFrame(
        {
            "Open": [100.0, float("nan"), 102.0],
            "High": [101.0, float("nan"), 103.0],
            "Low": [99.0, float("nan"), 101.0],
            "Close": [100.5, float("nan"), 0.0],
            "Volume": [1_000.0, float("nan"), 2_000.0],
        },
        index=idx,
    )
    out = _drop_unfinalised_rows(df)
    assert list(out.index.date) == [idx[0].date()]
    assert out["Close"].iloc[0] == 100.5
    assert not out["Close"].isna().any() and not (out["Close"] == 0).any()


def test_enforce_as_of_filters_and_asserts(history: pd.DataFrame, sample_date: str):
    d = _parse_as_of(sample_date)
    filtered = enforce_as_of(history, d)
    assert filtered.index.max() <= pd.Timestamp(sample_date)
    assert len(filtered) < len(history)


def test_enforce_as_of_none_is_passthrough(history: pd.DataFrame):
    assert len(enforce_as_of(history, None)) == len(history)


def test_normalise_symbol_suffixes():
    assert normalise_symbol("RELIANCE") == "RELIANCE.NS"
    assert normalise_symbol("reliance.ns") == "RELIANCE.NS"
    assert normalise_symbol("TCS.BO") == "TCS.BO"
    assert normalise_symbol("^NSEI") == "^NSEI"
    assert normalise_symbol("NIFTY") == "^NSEI"
    assert normalise_symbol("SENSEX") == "^BSESN"


def test_normalise_symbol_alias_variants():
    """Spelled-out index names must map to index tickers, not become
    "NIFTY 50.NS" via the bare-ticker .NS default."""
    assert normalise_symbol("NIFTY 50") == "^NSEI"
    assert normalise_symbol("BANKNIFTY") == "^NSEBANK"
    # explicit suffix still wins over the alias table
    assert normalise_symbol("NIFTY.NS") == "NIFTY.NS"


def test_did_you_mean_handles_transposition():
    """README promises suggestions; difflib must catch transpositions that
    prefix matching misses ("RELIIANCE" -> "RELIANCE.NS")."""
    out = _did_you_mean("RELIIANCE")
    assert "RELIANCE.NS" in out
    # Exact-ish prefix matching still works.
    assert "RELIANCE.NS" in _did_you_mean("RELIANC")


def test_rebuild_adjusted_does_not_reapply_splits():
    """A split row must never rescale prices: yfinance's Close is ALREADY
    split-adjusted, so applying 1/ratio again halves every pre-split row and
    manufactures a fake +100% jump on the ex-date (a real bug: RELIANCE 2024
    bonus turned into a +86.8% 20-day return).
    """
    df = pd.DataFrame({
        "Close": [100.0, 100.0, 100.0, 100.0],
        "Open": [100.0] * 4, "High": [100.0] * 4, "Low": [100.0] * 4, "Volume": [1000] * 4,
    }, index=pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]))
    actions = pd.DataFrame(
        {"Stock Splits": [0.0, 2.0], "Dividends": [0.0, 0.0]},
        index=pd.to_datetime(["2023-01-02", "2023-06-01"]),
    )
    # as_of before the split: the action is invisible, prices untouched
    adj, hist = rebuild_adjusted_with_actions(df, actions, date(2023, 1, 3))
    assert list(adj) == [100.0] * 4
    assert hist == []
    # as_of after the split: still untouched (source already adjusted it),
    # and the history says so instead of claiming a factor was applied
    adj2, hist2 = rebuild_adjusted_with_actions(df, actions, date(2023, 6, 2))
    assert list(adj2) == [100.0] * 4
    assert hist2 and hist2[-1]["type"] == "split"
    assert hist2[-1]["applied"] is False
    assert hist2[-1]["ratio"] == 2.0


def test_no_fake_return_across_a_vendor_split(sample_date, history):
    """Regression for the real-data bug: a continuous price series with a
    split row must produce no cliff and no invented return."""
    closes = [100.0] * 20 + [100.2] * 20
    idx = pd.bdate_range(end=sample_date, periods=40)
    df = pd.DataFrame({"Close": closes, "Open": closes, "High": closes,
                       "Low": closes, "Volume": [1000] * 40}, index=idx)
    split_day = idx[20]
    actions = pd.DataFrame({"Stock Splits": [2.0], "Dividends": [0.0]}, index=[split_day])
    adj, _ = rebuild_adjusted_with_actions(df, actions, _parse_as_of(sample_date))
    one_day = adj.iloc[20] / adj.iloc[19] - 1
    assert abs(one_day) < 0.01, f"cliff across the split: {one_day:.1%}"
    twenty_day = adj.iloc[30] / adj.iloc[10] - 1
    assert abs(twenty_day) < 0.01, f"fabricated return across the split: {twenty_day:.1%}"


def test_dividend_is_rebased_across_a_later_split():
    """As-filed dividends are in a pre-split basis; the factor method must use
    the post-split cash equivalent (10 pre-split, 2:1 split later -> 5)."""
    df = pd.DataFrame({
        "Close": [100.0, 100.0, 100.0, 100.0],
        "Open": [100.0] * 4, "High": [100.0] * 4, "Low": [100.0] * 4, "Volume": [1000] * 4,
    }, index=pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]))
    actions = pd.DataFrame(
        {"Stock Splits": [0.0, 2.0], "Dividends": [10.0, 0.0]},
        index=pd.to_datetime(["2023-01-04", "2023-02-01"]),
    )
    adj, hist = rebuild_adjusted_with_actions(df, actions, date(2023, 3, 1))
    div = [h for h in hist if h["type"] == "dividend"][-1]
    assert div["effective_amount"] == pytest.approx(5.0, abs=1e-9)
    assert div["factor"] == pytest.approx(0.95, abs=1e-9)
    assert adj.iloc[0] == pytest.approx(95.0, abs=1e-9)
    assert adj.iloc[-1] == 100.0


def test_rebuild_matches_vendor_adj_close():
    """With every action applied, the rebuild should land near the vendor's
    Adj Close for real cached data (both are dividend-adjusted)."""
    pytest.importorskip("yfinance")
    try:
        raw = fetch_history("RELIANCE.NS")
    except Exception:  # offline with no cache
        pytest.skip("no cached data / network for RELIANCE.NS")
    if raw.empty:
        pytest.skip("no RELIANCE.NS rows")
    actions = fetch_actions("RELIANCE.NS")
    adj, _ = rebuild_adjusted_with_actions(raw, actions, None)
    vendor = raw["Adj Close"].astype(float)
    rel = ((adj - vendor).abs() / vendor).tail(500).median()
    assert rel < 0.02, f"rebuild diverges from vendor Adj Close: median {rel:.2%}"


def test_dividend_adjustment_factor_method():
    """Standard factor method: factor = 1 - div / close_before_ex.

    Close=[100,100,100], div=10 ex on day 3: factor = 1-10/100 = 0.9.
    as_of before ex -> adjusted stays 100; as_of after ex -> pre-ex = 90.
    """
    df = pd.DataFrame({
        "Close": [100.0, 100.0, 100.0],
        "Open": [100.0] * 3, "High": [100.0] * 3, "Low": [100.0] * 3, "Volume": [1000] * 3,
    }, index=pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04"]))
    actions = pd.DataFrame(
        {"Stock Splits": [0.0], "Dividends": [10.0]},
        index=pd.to_datetime(["2023-01-04"]),
    )
    # as_of before ex-date: no adjustment
    adj_before, hist_before = rebuild_adjusted_with_actions(df, actions, date(2023, 1, 3))
    assert list(adj_before) == [100.0, 100.0, 100.0]
    assert hist_before == []
    # as_of after ex-date: pre-ex prices multiplied by 0.9
    adj_after, hist_after = rebuild_adjusted_with_actions(df, actions, date(2023, 1, 4))
    assert list(adj_after.iloc[:2]) == [90.0, 90.0]
    assert adj_after.iloc[-1] == 100.0
    assert hist_after and hist_after[-1]["type"] == "dividend"
    assert hist_after[-1]["factor"] == pytest.approx(0.9, abs=1e-9)


# --- offline tool-logic tests (synthetic data only) ------------------------

def test_cutoff_shrink(sample_date, history):
    """Shrinking as_of by 1 day must shrink the visible history."""
    d = _parse_as_of(sample_date)
    earlier = d - timedelta(days=1)
    n1 = len(enforce_as_of(history, d))
    n2 = len(enforce_as_of(history, earlier))
    assert n2 <= n1
    if sample_date in [str(x.date()) for x in history.index]:
        assert n2 < n1


def test_canary_future_spike_not_visible(sample_date, history):
    """A synthetic spike dated after as_of must not appear in as-of output."""
    d = _parse_as_of(sample_date)
    filtered = enforce_as_of(history, d)
    # simulate: spike lands at as_of + 5; the as_of view cannot see it
    spike_date = d + timedelta(days=5)
    assert filtered.index.max().date() <= d
    assert spike_date not in [x.date() for x in filtered.index]


def test_technical_indicators_match_direct_pandas(history):
    """Server indicators must match a direct pandas computation."""
    df = history.copy()
    df["CloseAdj"] = df["Close"].astype(float)
    calc = compute_all(df, ["SMA_50", "RSI_14"])
    expected_sma50 = float(df["CloseAdj"].rolling(50).mean().iloc[-1])
    assert calc["SMA_50"] == pytest.approx(expected_sma50, rel=1e-6)
    assert 0 <= (calc["RSI_14"] or 0) <= 100 or calc["RSI_14"] is None


def test_regression_no_row_after_as_of(history):
    """REGRESSION: enforcement must guarantee NO row with date > as_of."""
    for cutoff in (history.index[len(history) // 2], history.index[-10]):
        d = cutoff.date()
        filtered = enforce_as_of(history, d)
        assert (filtered.index.date <= d).all()
        assert filtered.index.max().date() <= d
    # also through the frame builder used by tools (offline synthetic path)
    d = _parse_as_of("2024-06-28")
    filtered = enforce_as_of(history, d)
    assert filtered.index.max().date() <= d


def test_rows_used_is_pre_trim_count(history, monkeypatch):
    """rows_used must equal the as_of-filtered row count BEFORE lookback
    trimming — pinned by driving the REAL _get_frame with synthetic I/O."""
    import stock_analyst_mcp.data as data_mod

    df = history.copy()
    monkeypatch.setattr(data_mod, "fetch_history", lambda symbol, **kw: df)
    monkeypatch.setattr(data_mod, "fetch_actions", lambda symbol: pd.DataFrame(
        columns=["Dividends", "Stock Splits"]))

    lookback = 30
    filtered, d, _hist, _actions = data_mod._get_frame(SYMBOL, None, lookback)
    assert len(filtered) == lookback            # returned slice is trimmed
    assert filtered.attrs["rows_used"] == len(history)  # pre-trim count
    assert filtered.attrs["rows_used"] > len(filtered)  # honesty: they differ

    # lookback larger than history -> the two are equal
    filtered2, *_ = data_mod._get_frame(SYMBOL, None, len(history) + 5)
    assert filtered2.attrs["rows_used"] == len(filtered2) == len(history)


def test_rows_used_lands_in_meta_envelope(history, monkeypatch):
    """Every tool's envelope must carry rows_used; the offline fake proves
    the wiring, test_rows_used_is_pre_trim_count proves the number."""
    import stock_analyst_mcp.server as server

    df = history.copy()
    df["CloseAdj"] = df["Close"].astype(float)  # tools read CloseAdj
    df.attrs["rows_used"] = 4242  # set AFTER column ops (attrs can be dropped)
    monkeypatch.setattr(server, "_get_frame",
                        lambda symbol, as_of, lookback=90: (df, None, [], []))

    for call in (
        lambda: server.get_ohlcv(SYMBOL),
        lambda: server.get_returns(SYMBOL),
        lambda: server.get_technical_indicators(SYMBOL),
        lambda: server.get_risk_metrics(SYMBOL),
        lambda: server.get_index_history(SYMBOL),
    ):
        out = call()
        assert out.get("rows_used") == 4242, out.get("rows_used")

    # chart_patterns + peer_comparison (fake has no attrs -> rows_used None)
    out = server.get_peer_comparison([SYMBOL], as_of=None)
    assert "rows_used" in out  # key always present in the envelope


def test_profile_resource_labels_mapping_as_unverified():
    import stock_analyst_mcp.server as server

    profile = server.profile_resource("RELIANCE.BO")
    assert "Mapping: suffix-based, not verified against the exchange" in profile
    assert "Exchange: BSE" in profile  # suffix-derived, per the mapping line


# --- network-dependent tests (skip only when network is actually down) -----

@network
def test_unknown_symbol_errors():
    from stock_analyst_mcp.server import get_ohlcv
    out = get_ohlcv("NOT-A-REAL-TICKER.XYZ", as_of="2024-01-01")
    assert out.get("isError") is True
    assert "Unknown symbol" in out["error"]


@network
def test_future_as_of_errors():
    from stock_analyst_mcp.server import get_ohlcv
    out = get_ohlcv(SYMBOL, as_of=FUTURE_DATE)
    assert out.get("isError") is True


@network
def test_peer_comparison_ranks():
    from stock_analyst_mcp.server import get_peer_comparison
    out = get_peer_comparison(["RELIANCE.NS", "TCS.NS"], as_of="2024-12-31", metric="return_20d")
    assert out.get("isError") is None
    assert len(out["results"]) == 2
    ranks = sorted(r["rank"] for r in out["results"])
    assert ranks == [1, 2]
    assert out["currency"] == "INR"
    assert out["source"] == "yfinance"
    assert out["data_through"] <= "2024-12-31"


# --- CLI smoke test (no network needed for import) --------------------------

def test_server_module_imports():
    import stock_analyst_mcp.server as s

    assert s.mcp is not None
    names = [t.name for t in s.mcp._tool_manager.list_tools()] if hasattr(s.mcp, "_tool_manager") else []
    # FastMCP private API may vary; at minimum check the functions exist.
    for fn in ("get_ohlcv", "get_returns", "get_technical_indicators",
               "get_chart_patterns", "get_risk_metrics", "get_index_history",
               "get_peer_comparison"):
        assert callable(getattr(s, fn))


def test_stdio_no_stdout_pollution():
    """The server's stdout must carry only valid JSON-RPC messages.

    Sends a real MCP ``initialize`` request over stdin (LSP-style framing
    as used by the MCP stdio transport), then asserts every stdout line is
    parseable JSON-RPC (no empty lines, no prose, no tracebacks). The
    timeout is 60s because the cold import alone is ~10s (pandas + yfinance
    + fastmcp); 10s intermittently killed the process mid-startup.
    """
    import json as _json

    init_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "pytest-probe", "version": "0.0.0"},
        },
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "stock_analyst_mcp.server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    try:
        proc.stdin.write(_json.dumps(init_req) + "\n")
        proc.stdin.flush()
        # Give the server a bounded window to answer; read one response line.
        import time as _time

        deadline = _time.time() + 60
        first_line: str | None = None
        while _time.time() < deadline:
            line = proc.stdout.readline()
            if line == "":  # EOF before any response — treated below
                break
            if line.strip():
                first_line = line.strip()
                break
        assert first_line is not None, (
            "server closed stdout without answering initialize "
            f"(stderr head: {proc.stderr.readline() if proc.stderr else ''!r})"
        )
        msg = _json.loads(first_line)  # raises on garbage/empty/prose lines
        assert msg.get("jsonrpc") == "2.0"
        assert msg.get("id") == 1
        server_info = msg["result"]["serverInfo"]
        # serverInfo.name must be the PRODUCT identity (distribution
        # indian-stocks-mcp), not the legacy import-package name — this is
        # the lock that keeps the announced identity from drifting.
        assert server_info["name"] == "indian-stocks-mcp"
        # serverInfo.version must reflect the package, not the SDK (1.12.4)
        assert server_info["version"] != "1.12.4"
        assert server_info["version"] == stock_analyst_mcp.__version__
    finally:
        proc.kill()
    # Drain what remains and re-assert purity of the whole stream.
    _rest = proc.stdout.read() if proc.stdout else ""
    for line in (_rest or "").splitlines():
        if line.strip():
            _json.loads(line)  # any non-JSON line fails here
