"""Performance metrics on a daily-return series / equity curve."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ANNUAL_VOL_SQRT, BUSINESS_DAYS_YEAR


def _clean(daily: pd.Series) -> pd.Series:
    return daily.dropna()


def ann_return(daily: pd.Series) -> float:
    return float(_clean(daily).mean() * BUSINESS_DAYS_YEAR)


def ann_vol(daily: pd.Series) -> float:
    return float(_clean(daily).std() * ANNUAL_VOL_SQRT)


def sharpe(daily: pd.Series) -> float:
    d = _clean(daily)
    s = d.std()
    return float(d.mean() / s * ANNUAL_VOL_SQRT) if s > 0 else float("nan")


def sortino(daily: pd.Series) -> float:
    d = _clean(daily)
    downside = d[d < 0].std()
    if not downside or np.isnan(downside):
        return float("nan")
    return float(d.mean() * BUSINESS_DAYS_YEAR / (downside * ANNUAL_VOL_SQRT))


def max_drawdown(equity: pd.Series) -> float:
    eq = equity.dropna()
    if eq.empty:
        return float("nan")
    return float((eq / eq.cummax() - 1.0).min())


def cagr(equity: pd.Series) -> float:
    eq = equity.dropna()
    if len(eq) < 2 or eq.iloc[-1] <= 0:
        return float("nan")
    years = len(eq) / BUSINESS_DAYS_YEAR
    return float(eq.iloc[-1] ** (1.0 / years) - 1.0)


def calmar(equity: pd.Series) -> float:
    mdd = max_drawdown(equity)
    c = cagr(equity)
    return float(c / abs(mdd)) if mdd and not np.isnan(mdd) and mdd != 0 else float("nan")


def skew(daily: pd.Series) -> float:
    return float(_clean(daily).skew())


def annual_turnover(turnover_daily: pd.Series) -> float:
    """Average daily traded-notional/capital × 256 → annualised turnover."""
    return float(_clean(turnover_daily).mean() * BUSINESS_DAYS_YEAR)


def summary(equity: pd.Series, daily: pd.Series, turnover: pd.Series | None = None) -> dict:
    out = {
        "sharpe": sharpe(daily),
        "ann_return": ann_return(daily),
        "ann_vol": ann_vol(daily),
        "max_drawdown": max_drawdown(equity),
        "cagr": cagr(equity),
        "calmar": calmar(equity),
        "sortino": sortino(daily),
        "skew": skew(daily),
        "n_days": int(_clean(daily).shape[0]),
    }
    if turnover is not None:
        out["ann_turnover"] = annual_turnover(turnover)
    return out
