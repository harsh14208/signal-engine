"""Volatility estimation — the denominator of every risk-adjusted forecast and
the basis of position sizing.

Carver's blended estimator: 70% recent exponentially-weighted vol + 30% long-run
average. The long-run term stops a single quiet/volatile patch from dominating
sizing, which is a common source of fragility.

Optionally blends in a GARCH(1,1)-t forward-vol estimate (`pip install -e .[garch]`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    ANNUAL_VOL_SQRT,
    VOL_EW_SPAN,
    VOL_GARCH_DIST,
    VOL_GARCH_HORIZON,
    VOL_GARCH_MIN_HISTORY,
    VOL_GARCH_REFIT_STEP,
    VOL_LONG_WEIGHT,
    VOL_MIN_PERIODS,
)

try:
    from arch import arch_model  # type: ignore[import-not-found]

    _GARCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _GARCH_AVAILABLE = False


def daily_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return prices.pct_change()


def ew_daily_vol(returns: pd.Series, span: int = VOL_EW_SPAN) -> pd.Series:
    """Exponentially-weighted daily-return standard deviation."""
    return returns.ewm(span=span, min_periods=VOL_MIN_PERIODS).std()


def _garch_daily_vol(
    returns: pd.Series,
    horizon: int = VOL_GARCH_HORIZON,
    min_history: int = VOL_GARCH_MIN_HISTORY,
    refit_step: int = VOL_GARCH_REFIT_STEP,
    dist: str = VOL_GARCH_DIST,
) -> pd.Series:
    """GARCH(1,1) forward vol, refit every `refit_step` bars.

    Uses returns[:i] to forecast bars i ... i+refit_step-1, so there is no
    lookahead. Returns NaN where the model cannot be fit.
    """
    if not _GARCH_AVAILABLE:
        return pd.Series(np.nan, index=returns.index)

    returns = pd.to_numeric(returns, errors="coerce").dropna()
    out = pd.Series(np.nan, index=returns.index)
    n = len(returns)
    if n < min_history + horizon + refit_step:
        return out

    scaled = returns.values * 100.0  # arch prefers O(1) scale
    for i in range(min_history, n - horizon, refit_step):
        hist = scaled[:i]
        try:
            res = arch_model(
                hist,
                vol="GARCH",
                p=1,
                q=1,
                dist=dist,
                rescale=False,
            ).fit(disp="off")
            fc = res.forecast(horizon=horizon, reindex=False)
            sigma_pct = np.sqrt(fc.variance.values[-1].mean())
        except Exception:
            continue
        if not (np.isfinite(sigma_pct) and sigma_pct > 0):
            continue
        end = min(i + refit_step, n)
        out.iloc[i:end] = sigma_pct / 100.0
    return out


def blended_daily_vol(
    returns: pd.Series,
    span: int = VOL_EW_SPAN,
    long_weight: float = VOL_LONG_WEIGHT,
    use_garch: bool = False,
    garch_weight: float = 0.0,
    garch_horizon: int = VOL_GARCH_HORIZON,
    garch_min_history: int = VOL_GARCH_MIN_HISTORY,
    garch_refit_step: int = VOL_GARCH_REFIT_STEP,
) -> pd.Series:
    """30% expanding long-run mean of EW vol + 70% recent EW vol, with optional GARCH blend.

    Warm-up observations where vol is not yet estimable are left as NaN rather
    than backfilled, avoiding a look-ahead leak.
    """
    recent = ew_daily_vol(returns, span)
    long_run = recent.expanding(min_periods=VOL_MIN_PERIODS).mean()
    ew_blend = long_weight * long_run + (1.0 - long_weight) * recent

    if not use_garch or garch_weight <= 0:
        return ew_blend

    garch = _garch_daily_vol(
        returns,
        horizon=garch_horizon,
        min_history=garch_min_history,
        refit_step=garch_refit_step,
    )

    combined = garch_weight * garch + (1.0 - garch_weight) * ew_blend
    combined = combined.where(garch.notna(), ew_blend)
    return combined


def annualise(daily_vol: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return daily_vol * ANNUAL_VOL_SQRT
