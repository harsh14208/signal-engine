"""Free carry proxies for the dormant carry rule.

Real carry needs futures term structure, but two clean legs are free right now:
  • bond carry from the yield-curve slope (FRED T10Y3M)
  • equity carry from trailing 12-month dividend yield (yfinance)

All fetchers cache results under `data/` so the engine stays deterministic offline
after the first run.
"""

from __future__ import annotations

import os
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd

from .config import Config
from .markets import Instrument, instrument_for

_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _cache_path(name: str) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, name)


def _fred_series(series_id: str) -> pd.Series:
    """Fetch a daily FRED series and return it as a timezone-naive Series."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
            df = pd.read_csv(resp, parse_dates=["observation_date"], index_col="observation_date")
    except Exception as exc:  # pragma: no cover - network failures handled upstream
        raise RuntimeError(f"Failed to fetch FRED {series_id}: {exc}") from exc
    s = df.iloc[:, 0].sort_index()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = series_id
    return s


def load_bond_carry(start: str | None = None, end: str | None = None) -> pd.Series:
    """Yield-curve slope (10Y - 3M) from FRED, as a decimal annual carry."""
    s = _fred_series("T10Y3M").dropna() / 100.0
    s.name = "bond_carry"
    start = pd.Timestamp(start or "1900-01-01")
    end = pd.Timestamp(end or datetime.now().strftime("%Y-%m-%d"))
    idx = pd.bdate_range(start=start, end=end)
    return s.reindex(idx, method="ffill").fillna(0.0)


def load_us_short_rate(start: str | None = None, end: str | None = None) -> pd.Series:
    """US 3-month Treasury yield (DGS3M0) from FRED, as a decimal."""
    s = _fred_series("DGS3MO").dropna() / 100.0
    s.name = "us_short_rate"
    start = pd.Timestamp(start or "1900-01-01")
    end = pd.Timestamp(end or datetime.now().strftime("%Y-%m-%d"))
    idx = pd.bdate_range(start=start, end=end)
    return s.reindex(idx, method="ffill").fillna(0.0)


def _load_dividends(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch historical dividends per symbol via yfinance and cache as parquet."""
    import yfinance as yf  # optional data dependency

    cache = _cache_path("dividends.parquet")
    if os.path.exists(cache):
        cached = pd.read_parquet(cache)
        # If cache already covers requested symbols and date range, reuse it.
        cached_syms = set(cached.columns)
        if cached_syms.issuperset(symbols):
            cached = cached.sort_index()
            cached.index = pd.to_datetime(cached.index).tz_localize(None)
            if cached.index.min() <= pd.Timestamp(start) and cached.index.max() >= pd.Timestamp(end):
                return cached.reindex(columns=symbols)

    start_dt = pd.Timestamp(start) - pd.DateOffset(years=2)
    frames: dict[str, pd.Series] = {}
    for sym in symbols:
        try:
            div = yf.Ticker(sym).dividends
            if div is None or div.empty:
                frames[sym] = pd.Series(dtype=float, name=sym)
                continue
            div = div.copy()
            div.index = pd.to_datetime(div.index).tz_localize(None)
            div = div[start_dt:end]
            frames[sym] = div
        except Exception:
            frames[sym] = pd.Series(dtype=float, name=sym)

    # Build daily panel and forward-fill.
    all_dates = pd.bdate_range(start=start_dt, end=end)
    panel = pd.DataFrame({sym: s.reindex(all_dates).fillna(0.0) for sym, s in frames.items()})
    panel.to_parquet(cache)
    return panel


def load_equity_carry(
    symbols: list[str],
    prices: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Trailing 12-month dividend yield for each symbol, aligned to `prices`."""
    if not symbols:
        return pd.DataFrame(index=prices.index)
    start = start or prices.index.min().strftime("%Y-%m-%d")
    end = end or prices.index.max().strftime("%Y-%m-%d")
    div_panel = _load_dividends(symbols, start, end)
    # Rolling 252-day sum of dividends, then divide by price.
    trailing = div_panel.rolling(window=252, min_periods=60).sum()
    carry = trailing / prices.reindex_like(trailing).replace(0.0, np.nan)
    return carry.reindex(prices.index).fillna(0.0)


def build_carry_panel(prices: pd.DataFrame, config: Config | None = None) -> pd.DataFrame:
    """Assemble a carry DataFrame aligned with `prices`.

    Bonds get the yield-curve slope; equities/real_estate/credit get trailing
    dividend yield; FX and commodities get zero.
    """
    config = config or Config()
    symbols = list(prices.columns)
    start = prices.index.min().strftime("%Y-%m-%d")
    end = prices.index.max().strftime("%Y-%m-%d")

    out = pd.DataFrame(0.0, index=prices.index, columns=symbols)

    expanded = getattr(config, "use_expanded_universe", False)
    bond_syms = [
        s
        for s in symbols
        if (instrument_for(s, expanded) or Instrument(s, "", "other")).carry_kind == "bond_slope"
    ]
    if bond_syms:
        bond_carry = load_bond_carry(start, end)
        bond_vals = bond_carry.reindex(out.index).ffill().fillna(0.0)
        out[bond_syms] = pd.DataFrame({s: bond_vals for s in bond_syms}, index=out.index).values

    equity_like_classes = {"equity", "real_estate", "credit"}
    equity_syms = [
        s
        for s in symbols
        if (instrument_for(s, expanded) or Instrument(s, "", "other")).asset_class in equity_like_classes
    ]
    if equity_syms:
        equity_carry = load_equity_carry(equity_syms, prices, start, end)
        out[equity_syms] = equity_carry.values

    return out
