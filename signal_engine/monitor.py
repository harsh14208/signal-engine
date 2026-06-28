"""No-broker monitoring: edge-decay detection + live-vs-backtest reconciliation.

The parent project's fatal gap was never reconciling live trading against its
backtest (81% of signals had a null field, sectors were silently unblocked,
calibration was silently off — all found late). This module is the cheap half you
can run BEFORE any broker:

  • `rolling_sharpe` / `edge_decay_report` — the strategy's own rolling 1-year
    Sharpe, with a decay alarm, so a dying edge is visible early.
  • `reconcile` — once you have live daily returns, measure whether they track the
    backtest (correlation, tracking error, drift). The live-vs-backtest agreement
    harness the parent never had.
"""

from __future__ import annotations

import pandas as pd

from .config import ANNUAL_VOL_SQRT, BUSINESS_DAYS_YEAR


def rolling_sharpe(daily: pd.Series, window: int = BUSINESS_DAYS_YEAR) -> pd.Series:
    """Annualised Sharpe over a trailing `window` (default ~1 year)."""
    d = daily.dropna()
    mean = d.rolling(window).mean()
    std = d.rolling(window).std()
    return (mean / std * ANNUAL_VOL_SQRT).rename("rolling_sharpe")


def edge_decay_report(
    daily: pd.Series, window: int = BUSINESS_DAYS_YEAR, alarm_floor: float = 0.0
) -> dict:
    """Summarise the rolling-Sharpe path and raise an alarm if the latest dips
    below `alarm_floor` — a cheap, honest 'is the edge still alive' monitor."""
    rs = rolling_sharpe(daily, window).dropna()
    if rs.empty:
        return {"insufficient": True}
    current = float(rs.iloc[-1])
    return {
        "window": window,
        "current": current,
        "median": float(rs.median()),
        "min": float(rs.min()),
        "max": float(rs.max()),
        "pct_windows_below_zero": float((rs < 0).mean()),
        "alarm_floor": alarm_floor,
        "alarm": current < alarm_floor,
    }


def reconcile(live_returns: pd.Series, backtest_returns: pd.Series) -> dict:
    """Align live vs backtest daily returns on shared dates and score the match.

    Until you have live data this is exercised in tests; once live, feed it the
    realised daily returns to catch slippage / divergence early.
    """
    df = pd.concat([live_returns.rename("live"), backtest_returns.rename("bt")], axis=1).dropna()
    if len(df) < 20:
        return {"insufficient": True, "n": int(len(df))}
    diff = df["live"] - df["bt"]
    corr = float(df["live"].corr(df["bt"]))
    tracking_error = float(diff.std() * ANNUAL_VOL_SQRT)  # annualised
    drift = float(diff.mean() * BUSINESS_DAYS_YEAR)  # annualised mean slippage
    return {
        "n": int(len(df)),
        "corr": corr,
        "tracking_error": tracking_error,
        "drift": drift,
        "aligned": corr > 0.80 and tracking_error < 0.05,
    }
