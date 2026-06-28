"""Empirical expanding-window forecast scalars.

The published EWMAC/breakout scalars are calibrated on futures. On real ETFs the
realised mean |forecast| often drifts above 10, which is one reason the
pre-governor book ran hot. This module estimates each rule's scalar from an
expanding window of the cached data with no lookahead, then scales the forecast
back to the target mean |forecast|.
"""

from __future__ import annotations

import json
import os

import pandas as pd

from .config import AVG_ABS_FORECAST, Config
from .rules import trend_forecasts
from .volatility import blended_daily_vol, daily_returns

_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_SCALAR_CACHE = os.path.join(_CACHE_DIR, "rule_scalars.json")


def _dynamic_scalar(forecast: pd.Series, target: float = AVG_ABS_FORECAST) -> pd.Series:
    """Time-varying scalar that forces an expanding mean |forecast| ≈ target."""
    mean_abs = forecast.abs().expanding(min_periods=20).mean()
    scalar = target / mean_abs.replace(0.0, 1.0)
    return scalar.shift(1).fillna(1.0)


def empirical_scalars(prices: pd.DataFrame, config: Config | None = None) -> dict[str, pd.Series]:
    """Return a dict of {rule_name: scalar_series} for dynamic rescaling."""
    config = config or Config()
    returns = daily_returns(prices)
    out: dict[str, pd.Series] = {}
    for sym in prices.columns:
        dvol = blended_daily_vol(
            returns[sym],
            use_garch=config.use_garch_vol,
            garch_weight=config.garch_weight,
            garch_horizon=config.garch_horizon,
            garch_min_history=config.garch_min_history,
            garch_refit_step=config.garch_refit_step,
        )
        forecasts = trend_forecasts(
            prices[sym],
            dvol,
            config.ewmac_speeds,
            config.breakout_spans if config.use_breakout else (),
        )
        for name, fc in forecasts.items():
            scalar = _dynamic_scalar(fc)
            if name not in out:
                out[name] = []
            out[name].append(scalar)

    # Average the per-instrument scalars rule-wise (they share the same index).
    averaged: dict[str, pd.Series] = {}
    for name, parts in out.items():
        df = pd.concat(parts, axis=1)
        averaged[name] = df.mean(axis=1).fillna(1.0)
    return averaged


def cache_scalars(scalars: dict[str, pd.Series]) -> None:
    """Write the mean scalar per rule to a JSON cache for reproducibility."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    summary = {name: float(s.dropna().mean()) for name, s in scalars.items()}
    with open(_SCALAR_CACHE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def load_cached_scalar_means() -> dict[str, float]:
    """Read the cached mean scalars if available."""
    if not os.path.exists(_SCALAR_CACHE):
        return {}
    with open(_SCALAR_CACHE, "r", encoding="utf-8") as f:
        return json.load(f)
