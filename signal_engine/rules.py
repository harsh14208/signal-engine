"""Trading rules → vol-normalised forecasts in Carver units (mean |f| ≈ 10,
capped ±20).

A forecast is a *risk-adjusted* view: how strong is the signal relative to the
instrument's own volatility. Normalising by vol is what lets a gold position and
a bond position speak the same language and be combined.

Rules implemented:
  • EWMAC  — exponentially-weighted moving-average crossover (the canonical trend
             rule), at multiple speeds.
  • Breakout — position within an N-day high/low channel (a second, weakly
             correlated trend family).
  • Carry  — risk-adjusted expected return from holding (term-structure driven).
  • Acceleration — trend curvature (fast EWMAC minus slow EWMAC).
  • Cross-sectional momentum — relative strength rank across the panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    AVG_ABS_FORECAST,
    BREAKOUT_SCALARS,
    CARRY_SCALAR,
    EWMAC_SCALARS,
    FORECAST_CAP,
)


def _cap(forecast: pd.Series, cap: float = FORECAST_CAP) -> pd.Series:
    return forecast.clip(-cap, cap)


def ewma(prices: pd.Series, span: int) -> pd.Series:
    return prices.ewm(span=span, min_periods=2).mean()


def ewmac_forecast(
    prices: pd.Series,
    daily_return_vol: pd.Series,
    fast: int,
    slow: int,
    scalar: float | None = None,
    cap: float = FORECAST_CAP,
) -> pd.Series:
    """Crossover normalised by price volatility, scaled to mean |f| ≈ 10.

    Uses the published constant scalar by default (an empirical per-series scalar
    would peek at the whole sample = lookahead).
    """
    price_vol = (daily_return_vol * prices).replace(0.0, np.nan)
    raw = (ewma(prices, fast) - ewma(prices, slow)) / price_vol
    if scalar is None:
        scalar = EWMAC_SCALARS.get((fast, slow))
        if scalar is None:  # speed not in the published table → empirical fallback
            scalar = AVG_ABS_FORECAST / raw.abs().mean()
    return _cap(raw * scalar, cap)


def breakout_forecast(
    prices: pd.Series, span: int, scalar: float | None = None, cap: float = FORECAST_CAP
) -> pd.Series:
    """Smoothed position within the rolling [min, max] channel, in [-1, 1]-ish,
    scaled to forecast units."""
    roll_max = prices.rolling(span, min_periods=span // 2).max()
    roll_min = prices.rolling(span, min_periods=span // 2).min()
    roll_mean = 0.5 * (roll_max + roll_min)
    width = (roll_max - roll_min).replace(0.0, np.nan)
    raw = (prices - roll_mean) / (0.5 * width)
    raw = raw.ewm(span=max(span // 4, 1), min_periods=2).mean()
    if scalar is None:
        scalar = BREAKOUT_SCALARS.get(span, AVG_ABS_FORECAST / raw.abs().mean())
    # raw is ~[-1, 1]; the published breakout scalar (~30) lifts mean|f| to ≈10.
    return _cap(raw * scalar, cap)


def carry_forecast(
    annualised_carry: pd.Series,
    annual_return_vol: pd.Series,
    scalar: float = CARRY_SCALAR,
    cap: float = FORECAST_CAP,
) -> pd.Series:
    """Risk-adjusted carry: expected annualised carry return / annualised vol.

    `annualised_carry` must be a real term-structure-derived series (see
    data.bond_carry_proxy / README §carry). Positive → long is paid to hold.
    """
    raw = annualised_carry / annual_return_vol.replace(0.0, np.nan)
    return _cap(raw * scalar, cap)


def acceleration_forecast(
    prices: pd.Series,
    daily_return_vol: pd.Series,
    fast_pair: tuple[int, int] = (8, 32),
    slow_pair: tuple[int, int] = (16, 64),
    cap: float = FORECAST_CAP,
) -> pd.Series:
    """Trend curvature: fast EWMAC minus slow EWMAC.

    Each leg is already scaled to mean |f| ≈ 10, so the difference is a
    naturally normalised acceleration signal.
    """
    fast = ewmac_forecast(prices, daily_return_vol, fast_pair[0], fast_pair[1], cap=cap)
    slow = ewmac_forecast(prices, daily_return_vol, slow_pair[0], slow_pair[1], cap=cap)
    return (fast - slow).clip(-cap, cap)


def cross_sectional_momentum_forecast(
    prices: pd.DataFrame, lookback: int = 64, cap: float = FORECAST_CAP
) -> pd.DataFrame:
    """Relative momentum: rank recent total returns across the panel.

    Percentile ranks per day are mapped linearly to [-cap, +cap]. The resulting
    forecast is cross-sectionally neutral (zero mean across instruments each day)
    and has a fixed cross-sectional mean |f| of cap/2 ≈ 10.
    """
    mom = prices.pct_change(lookback)
    rank = mom.rank(axis=1, pct=True)
    forecast = (rank - 0.5) * 2.0 * cap
    return forecast


def trend_forecasts(
    prices: pd.Series,
    daily_return_vol: pd.Series,
    ewmac_speeds,
    breakout_spans=(),
) -> dict[str, pd.Series]:
    """All trend forecasts for one instrument, keyed by rule name."""
    out: dict[str, pd.Series] = {}
    for fast, slow in ewmac_speeds:
        out[f"ewmac_{fast}_{slow}"] = ewmac_forecast(prices, daily_return_vol, fast, slow)
    for span in breakout_spans:
        out[f"breakout_{span}"] = breakout_forecast(prices, span)
    return out
