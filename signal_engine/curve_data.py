"""Synthetic curve instruments built from free FRED Treasury yields.

The 2s10s steepener is a long-short macro factor, not a tradable ETF. It is
added as a single synthetic instrument so the engine's vol-targeting, IDM and
rules can size it like any other bet.
"""

from __future__ import annotations

import os

import pandas as pd

from .carry_data import load_treasury_curve

_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _cache_path(name: str) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, name)


def load_2s10s_steepener(
    start: str | None = None,
    end: str | None = None,
    scale: float = 1.0,
) -> pd.Series:
    """Return a synthetic price series for a long-2Y/short-10Y curve steepener.

    Daily return is the change in the 10Y-2Y spread (in decimal) times `scale`.
    The price index starts at 100 and is cumulated from those returns, so the
    engine's vol-targeting can size it normally.
    """
    start = start or "2007-01-01"
    end = end or pd.Timestamp.now().strftime("%Y-%m-%d")
    cache = _cache_path("ust2s10s.parquet")

    if os.path.exists(cache):
        cached = pd.read_parquet(cache)
        cached.index = pd.to_datetime(cached.index).tz_localize(None)
        if cached.index.min() <= pd.Timestamp(start) and cached.index.max() >= pd.Timestamp(end):
            s = cached.loc[start:end].iloc[:, 0]
            s.name = "UST2S10S"
            return s

    curve = load_treasury_curve(start, end)
    spread = curve["DGS10"] - curve["DGS2"]
    daily_ret = spread.diff().fillna(0.0) * scale
    price = 100.0 * (1.0 + daily_ret).cumprod()
    price.name = "UST2S10S"
    price.to_frame().to_parquet(cache)
    return price


def load_curve_instruments(
    prices: pd.DataFrame,
    steepener: bool = True,
    scale: float = 1.0,
) -> pd.DataFrame:
    """Return a DataFrame of synthetic curve-instrument prices aligned to `prices`."""
    start = prices.index.min().strftime("%Y-%m-%d")
    end = prices.index.max().strftime("%Y-%m-%d")
    out: dict[str, pd.Series] = {}
    if steepener:
        out["UST2S10S"] = load_2s10s_steepener(start, end, scale=scale)
    if not out:
        return pd.DataFrame(index=prices.index)
    df = pd.DataFrame(out)
    df = df.reindex(prices.index).ffill().bfill()
    return df
