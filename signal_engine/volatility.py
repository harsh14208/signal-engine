"""Volatility estimation — the denominator of every risk-adjusted forecast and
the basis of position sizing.

Carver's blended estimator: 70% recent exponentially-weighted vol + 30% long-run
average. The long-run term stops a single quiet/volatile patch from dominating
sizing, which is a common source of fragility.
"""

from __future__ import annotations

import pandas as pd

from .config import ANNUAL_VOL_SQRT, VOL_EW_SPAN, VOL_LONG_WEIGHT, VOL_MIN_PERIODS


def daily_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return prices.pct_change()


def ew_daily_vol(returns: pd.Series, span: int = VOL_EW_SPAN) -> pd.Series:
    """Exponentially-weighted daily-return standard deviation."""
    return returns.ewm(span=span, min_periods=VOL_MIN_PERIODS).std()


def blended_daily_vol(
    returns: pd.Series,
    span: int = VOL_EW_SPAN,
    long_weight: float = VOL_LONG_WEIGHT,
) -> pd.Series:
    """30% expanding long-run mean of EW vol + 70% recent EW vol."""
    recent = ew_daily_vol(returns, span)
    long_run = recent.expanding(min_periods=VOL_MIN_PERIODS).mean()
    blended = long_weight * long_run + (1.0 - long_weight) * recent
    # Floor: never allow a zero/NaN vol to blow up sizing.
    return blended.bfill()


def annualise(daily_vol: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return daily_vol * ANNUAL_VOL_SQRT
