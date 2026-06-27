"""Position sizing — turn a forecast into a number of contracts/shares at a
constant risk target.

Two ideas do all the work:

  • Volatility targeting: every instrument is sized so that, at the average
    forecast (10), it contributes the same dollar risk. A 30%-vol commodity and
    a 6%-vol bond therefore take very different notional sizes but equal RISK.

  • Instrument Diversification Multiplier (IDM): a basket of weakly-correlated
    instruments has far lower vol than each leg, so the whole book is scaled up
    by IDM = 1/sqrt(w'·Ρ·w) to actually hit the portfolio vol target. THIS is
    where diversification turns into return — it is the mathematical statement
    of "many uncorrelated bets stack."
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AVG_ABS_FORECAST, IDM_CAP


def estimate_idm(returns: pd.DataFrame, weights: dict[str, float], cap: float = IDM_CAP) -> float:
    """IDM from the instrument-return correlation matrix."""
    cols = [c for c in weights if c in returns.columns]
    if len(cols) <= 1:
        return 1.0
    corr = returns[cols].corr().fillna(0.0).values
    np.fill_diagonal(corr, 1.0)
    w = np.array([weights[c] for c in cols])
    port_var = float(w @ corr @ w)
    if port_var <= 0:
        return 1.0
    return float(min(1.0 / np.sqrt(port_var), cap))


def position_units(
    forecast: pd.Series,
    price: pd.Series,
    annual_return_vol: pd.Series,
    capital: float,
    vol_target: float,
    instrument_weight: float,
    idm: float,
    multiplier: float = 1.0,
) -> pd.Series:
    """Number of units (contracts/shares) to hold.

        units = (forecast/10) · (capital·vol_target·weight·IDM) / (σ_annual·price·mult)

    The numerator is this instrument's dollar risk budget; the denominator is the
    dollar risk of holding ONE unit for a year.
    """
    risk_budget = capital * vol_target * instrument_weight * idm
    risk_per_unit = (annual_return_vol * price * multiplier).replace(0.0, np.nan)
    units = (forecast / AVG_ABS_FORECAST) * risk_budget / risk_per_unit
    return units.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def apply_buffer(units: pd.Series, fraction: float) -> pd.Series:
    """No-trade buffer to cut turnover: only move to the new target when it is
    more than `fraction` × (typical position size) away from the current hold.

    The band uses an EXPANDING mean of |units| (no lookahead)."""
    if fraction <= 0:
        return units
    band = (units.abs().expanding(min_periods=20).mean() * fraction).bfill().fillna(0.0)
    vals = units.to_numpy()
    bands = band.to_numpy()
    held = np.empty_like(vals)
    cur = 0.0
    for i in range(len(vals)):
        target = vals[i]
        if np.isnan(target):
            held[i] = cur
            continue
        if abs(target - cur) > bands[i]:
            cur = target
        held[i] = cur
    return pd.Series(held, index=units.index)
