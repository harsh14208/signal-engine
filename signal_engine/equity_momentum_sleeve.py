"""Synthetic S&P 500 cross-sectional momentum sleeve from the parent's PIT data.

Builds a dollar-neutral top/bottom-decile momentum portfolio using the cached
S&P 500 OHLCV and point-in-time membership from the sibling
`TradingRecommendationSystem` repo. The result is a single synthetic price series
(`SP500_XSMOM`) that can be merged into the engine's price panel.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

_PARENT_DATA = Path("/Users/harshv.singh/TradingRecommendationSystem/backend/data")
_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _cache_path(name: str) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, name)


def _load_membership(start: str, end: str) -> pd.DataFrame:
    """Return a boolean membership DataFrame (date × ticker) from PIT files."""
    csv_path = _PARENT_DATA / "sp500_ticker_start_end.csv"
    json_path = _PARENT_DATA / "sp500_historical_constituents.json"

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    idx = pd.bdate_range(start=start_dt, end=end_dt)

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif json_path.exists():
        import json

        data = json.loads(json_path.read_text())
        rows = []
        for ticker, intervals in data.items():
            if ticker.startswith("_"):
                continue
            for start_str, end_str in intervals:
                rows.append({"ticker": ticker, "start_date": start_str, "end_date": end_str})
        df = pd.DataFrame(rows)
    else:
        raise FileNotFoundError("No S&P 500 PIT membership file found in parent data")

    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce").fillna(pd.Timestamp("2100-01-01"))

    # Restrict to tickers that were ever members during the target window.
    ever = df[(df["start_date"] <= end_dt) & (df["end_date"] >= start_dt)]
    tickers = sorted(ever["ticker"].unique())

    membership = pd.DataFrame(False, index=idx, columns=tickers)
    for _, row in ever.iterrows():
        mask = (membership.index >= row["start_date"]) & (membership.index <= row["end_date"])
        if row["ticker"] in membership.columns:
            membership.loc[mask, row["ticker"]] = True
    return membership


def _latest_ohlcv_path(ticker: str) -> Path | None:
    """Pick the OHLCV cache file for `ticker` with the latest end date."""
    files = list((_PARENT_DATA / "cache_ohlcv").glob(f"{ticker}_*_1d_adjTrue.csv"))
    if not files:
        return None
    # Filename: {TICKER}_{start}_{end}_1d_adjTrue.csv
    def _end(f: Path) -> str:
        parts = f.stem.split("_")
        return parts[2] if len(parts) >= 3 else ""
    return max(files, key=lambda f: _end(f))


def _load_close(ticker: str, start: str, end: str) -> pd.Series | None:
    """Load adjusted close for a single ticker from the parent's cache."""
    path = _latest_ohlcv_path(ticker)
    if path is None:
        return None
    try:
        df = pd.read_csv(
            path,
            skiprows=3,
            names=["Date", "Open", "High", "Low", "Close", "Volume"],
            parse_dates=["Date"],
            index_col="Date",
        )
    except Exception:
        return None
    s = df["Close"].sort_index()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.loc[start:end]


def build_equity_momentum_sleeve(
    start: str | None = None,
    end: str | None = None,
    lookback: int = 252,
    rebalance: int = 21,
    decile: float = 0.10,
    min_stocks: int = 20,
) -> pd.Series:
    """Return a synthetic price series for an S&P 500 momentum L/S sleeve.

    Long the top `decile` of S&P 500 members by 12-1 month momentum,
    short the bottom decile, equal-weighted, rebalanced every `rebalance` days.
    """
    start = start or "2007-01-01"
    end = end or pd.Timestamp.now().strftime("%Y-%m-%d")
    cache = _cache_path("sp500_xsmom.parquet")

    if os.path.exists(cache):
        cached = pd.read_parquet(cache)
        cached.index = pd.to_datetime(cached.index).tz_localize(None)
        if cached.index.min() <= pd.Timestamp(start) and cached.index.max() >= pd.Timestamp(end):
            s = cached.loc[start:end].iloc[:, 0]
            s.name = "SP500_XSMOM"
            return s

    membership = _load_membership(start, end)
    idx = membership.index
    tickers = list(membership.columns)

    close = pd.DataFrame(np.nan, index=idx, columns=tickers)
    for ticker in tickers:
        s = _load_close(ticker, start, end)
        if s is not None and len(s.dropna()) >= lookback + rebalance:
            close[ticker] = s.reindex(idx)

    # Mask non-membership days.
    close = close.where(membership)

    # Momentum signal: 12-1 month momentum.
    mom = close.shift(21) / close.shift(lookback) - 1.0

    # Rebalance dates: every `rebalance` days after warmup.
    warmup = lookback + rebalance
    reb_dates = idx[warmup::rebalance]

    daily_ret = pd.Series(0.0, index=idx)

    for i, d in enumerate(reb_dates):
        next_d = reb_dates[i + 1] if i + 1 < len(reb_dates) else idx[-1]
        valid = mom.loc[d].dropna()
        valid = valid[valid.index.isin(close.columns)]
        if len(valid) < 2 * min_stocks:
            continue
        n = max(1, int(len(valid) * decile))
        longs = valid.nlargest(n).index.tolist()
        shorts = valid.nsmallest(n).index.tolist()

        period_idx = idx[(idx > d) & (idx <= next_d)]
        if len(period_idx) == 0:
            continue
        long_rets = close[longs].pct_change().reindex(period_idx).clip(-0.30, 0.30)
        short_rets = close[shorts].pct_change().reindex(period_idx).clip(-0.30, 0.30)
        long_ret = long_rets.mean(axis=1)
        short_ret = short_rets.mean(axis=1)
        ls = (long_ret - short_ret).clip(-0.50, 0.50)
        daily_ret.loc[period_idx] = ls

    # Cumulate to a price index.
    price = 100.0 * (1.0 + daily_ret).cumprod()
    price.name = "SP500_XSMOM"
    price.to_frame().to_parquet(cache)
    return price
