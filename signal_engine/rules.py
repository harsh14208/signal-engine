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
    VOL_MIN_PERIODS,
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


def _lead_lag_adjacency(window: pd.DataFrame, lag: int, top_k: int) -> np.ndarray:
    """Directed leader→follower weight matrix from lagged cross-correlation.

    ``W[l, m]`` is the (renormalised) strength with which instrument *l*'s past
    return predicts instrument *m*'s current return. Only positive correlations
    are kept; each follower keeps its ``top_k`` leaders, weights sum to 1.
    """
    cols = list(window.columns)
    n = len(cols)
    lagged = window.shift(lag)
    w = np.zeros((n, n))
    for j, m in enumerate(cols):  # follower m (column j)
        ym = window[m]
        scores: list[tuple[int, float]] = []
        for i, leader in enumerate(cols):
            if i == j:
                continue
            pair = pd.concat([lagged[leader], ym], axis=1).dropna()
            if len(pair) < 20:
                continue
            c = pair.iloc[:, 0].corr(pair.iloc[:, 1])
            if pd.notna(c) and c > 0:
                scores.append((i, float(c)))
        if not scores:
            continue
        scores.sort(key=lambda kv: kv[1], reverse=True)
        top = scores[:top_k]
        total = sum(s for _, s in top)
        if total > 0:
            for i, s in top:
                w[i, j] = s / total
    return w


def network_momentum_forecast(
    prices: pd.DataFrame,
    returns: pd.DataFrame | None = None,
    speed: tuple[int, int] = (32, 128),
    lookback: int = 256,
    lag: int = 1,
    rebal: int = 63,
    top_k: int = 3,
    cap: float = FORECAST_CAP,
) -> pd.DataFrame:
    """Network ("follow the leader") momentum — a price-only cross-market signal.

    Each instrument's forecast is the adjacency-weighted sum of its *leaders'*
    time-series momentum, where the lead-lag graph is learned from trailing
    lagged cross-correlations (arXiv 2501.07135). This is a genuinely new,
    weakly-correlated return stream built from prices you already hold — a fit
    for the "stack uncorrelated bets" thesis. Opt-in; validate vs placebo/CPCV
    before promoting.

    Causal: the adjacency at each rebal point uses only the trailing ``lookback``
    window, and each leader's momentum uses only past prices.
    """
    prices = prices.sort_index()
    if returns is None:
        returns = prices.pct_change()
    cols = list(prices.columns)
    n = len(cols)
    out = pd.DataFrame(0.0, index=prices.index, columns=cols)
    if n < 2:
        return out

    fast, slow = speed
    base = pd.DataFrame(
        {
            c: ewmac_forecast(
                prices[c], returns[c].ewm(span=32, min_periods=VOL_MIN_PERIODS).std(), fast, slow
            )
            for c in cols
        }
    )
    base_arr = base.to_numpy()
    ret = returns[cols]
    t = len(prices)
    start = max(lookback, slow)
    if start >= t:
        return out

    rebal_points = list(range(start, t, max(1, rebal)))
    for seg, r0 in enumerate(rebal_points):
        r1 = rebal_points[seg + 1] if seg + 1 < len(rebal_points) else t
        window = ret.iloc[max(0, r0 - lookback) : r0]
        w = _lead_lag_adjacency(window, lag, top_k)
        out.iloc[r0:r1] = base_arr[r0:r1] @ w
    return out.clip(-cap, cap)


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
