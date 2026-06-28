"""Combine multiple rule forecasts for one instrument into a single capped
forecast, applying the Forecast Diversification Multiplier (FDM).

Why FDM: averaging positively-correlated forecasts shrinks the result toward
zero (a diversified average has lower variance). FDM = 1/sqrt(w'·C·w) scales the
combined forecast back up to the target average of 10, capped for safety.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FDM_CAP, FORECAST_CAP


def equal_weights(keys) -> dict[str, float]:
    keys = list(keys)
    w = 1.0 / len(keys) if keys else 0.0
    return {k: w for k in keys}


def estimate_fdm(
    forecast_corr: pd.DataFrame, weights: dict[str, float], cap: float = FDM_CAP
) -> float:
    """FDM from a rule×rule forecast-correlation matrix (estimate pooled across
    instruments for stability)."""
    cols = list(weights)
    if len(cols) <= 1:
        return 1.0
    c = forecast_corr.reindex(index=cols, columns=cols).fillna(0.0).to_numpy().copy()
    np.fill_diagonal(c, 1.0)
    w = np.array([weights[k] for k in cols])
    port_var = float(w @ c @ w)
    if port_var <= 0:
        return 1.0
    return float(min(1.0 / np.sqrt(port_var), cap))


def combine_instrument(
    forecasts: dict[str, pd.Series],
    weights: dict[str, float] | None = None,
    fdm: float | pd.Series = 1.0,
    cap: float = FORECAST_CAP,
) -> pd.Series:
    """Weighted sum of rule forecasts × FDM, capped at ±`cap`.

    `fdm` may be a scalar or a daily Series for expanding-window calibration.
    """
    if weights is None:
        weights = equal_weights(forecasts.keys())
    df = pd.DataFrame(forecasts)
    combined = sum(df[k] * weights.get(k, 0.0) for k in df.columns)
    if isinstance(fdm, pd.Series):
        combined = combined * fdm.reindex(combined.index).fillna(1.0)
    else:
        combined = combined * fdm
    return combined.clip(-cap, cap)


def pooled_rule_correlation(per_instrument: dict[str, dict[str, pd.Series]]) -> pd.DataFrame:
    """Rule×rule forecast correlation pooled across all instruments.

    per_instrument: {symbol: {rule_name: forecast_series}}.
    """
    stacked: dict[str, list[pd.Series]] = {}
    for _sym, rules in per_instrument.items():
        for name, series in rules.items():
            stacked.setdefault(name, []).append(series.reset_index(drop=True))
    cols = {name: pd.concat(parts, ignore_index=True) for name, parts in stacked.items()}
    return pd.DataFrame(cols).corr()
