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

from .config import ANNUAL_VOL_SQRT, AVG_ABS_FORECAST, IDM_CAP


def estimate_idm(returns: pd.DataFrame, weights: dict[str, float], cap: float = IDM_CAP) -> float:
    """IDM from the instrument-return correlation matrix."""
    cols = [c for c in weights if c in returns.columns]
    if len(cols) <= 1:
        return 1.0
    corr = returns[cols].corr().fillna(0.0).to_numpy().copy()
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
    instrument_weight: float | pd.Series,
    idm: float | pd.Series,
    multiplier: float = 1.0,
) -> pd.Series:
    """Number of units (contracts/shares) to hold.

        units = (forecast/10) · (capital·vol_target·weight·IDM) / (σ_annual·price·mult)

    The numerator is this instrument's dollar risk budget; the denominator is the
    dollar risk of holding ONE unit for a year.  `instrument_weight` and `idm`
    may be scalars or Series aligned with `forecast`.
    """
    risk_budget = capital * vol_target * instrument_weight * idm
    risk_per_unit = (annual_return_vol * price * multiplier).replace(0.0, np.nan)
    units = (forecast / AVG_ABS_FORECAST) * risk_budget / risk_per_unit
    return units.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def apply_crypto_risk_cap(
    units: pd.DataFrame,
    prices: pd.DataFrame,
    annual_return_vols: pd.DataFrame,
    config,
) -> pd.DataFrame:
    """Scale crypto positions so no single crypto instrument exceeds the configured
    fraction of portfolio risk target.

    The cap is applied to standalone dollar risk (|units|·price·σ_annual) as a
    fraction of the portfolio risk budget (capital·vol_target).  This is a
    conservative, causal guard; it does not account for diversification, but it
    prevents a single high-vol crypto name from dominating the book.
    """
    if not config.use_crypto or config.crypto_max_risk_weight is None:
        return units
    from .markets import instrument_for

    out = units.copy()
    budget = config.capital * config.vol_target
    if budget <= 0:
        return out
    for sym in units.columns:
        inst = instrument_for(sym)
        if inst is None or inst.asset_class != "crypto":
            continue
        risk = (out[sym].abs() * prices[sym] * inst.multiplier * annual_return_vols[sym]).abs()
        frac = risk / budget
        above = frac > config.crypto_max_risk_weight
        if above.any():
            scale = pd.Series(np.where(above, config.crypto_max_risk_weight / frac, 1.0), index=units.index)
            # Forward-fill scale so a cap, once applied, doesn't flip back and forth
            # on noise; only relax when the unconstrained position would be below cap.
            out[sym] = out[sym] * scale
    return out


def apply_buffer(units: pd.Series, fraction: float) -> pd.Series:
    """No-trade buffer to cut turnover: only move to the new target when it is
    more than `fraction` × (typical position size) away from the current hold.

    The band uses an EXPANDING mean of |units| (no lookahead)."""
    if fraction <= 0:
        return units
    band = (units.abs().expanding(min_periods=20).mean() * fraction).fillna(0.0)
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


def vol_governor(
    daily_returns: pd.Series,
    target_vol: float,
    span: int = 32,
    lo: float = 0.20,
    hi: float = 2.50,
    smooth: int | None = None,
) -> pd.Series:
    """Leverage multiplier that drags realised vol toward `target_vol`.

        governor_t = clip(target_vol / trailing_realised_vol_{t-1}, [lo, hi])

    The trailing realised vol is an EW estimate of the ungoverned strategy's own
    daily returns, LAGGED one day (`.shift(1)`) so the multiplier applied on day t
    uses only information available at t-1 — no lookahead. Clamped to avoid
    blowing up leverage in unusually calm regimes.

    If `smooth` is given, the raw multiplier is passed through a short EWMA
    (using only current and past values, because the input is already lagged)
    to reduce the turnover generated by noisy daily changes.
    """
    trailing = daily_returns.ewm(span=span, min_periods=20).std() * ANNUAL_VOL_SQRT
    gov = (target_vol / trailing.replace(0.0, np.nan)).shift(1)
    if smooth is not None and smooth > 1:
        gov = gov.ewm(span=smooth, min_periods=1).mean()
    return gov.clip(lo, hi).fillna(1.0)


def trend_strength_overlay(
    forecasts: pd.DataFrame,
    window: int = 63,
    threshold: float = 0.25,
    scale: float = 0.70,
) -> pd.Series:
    """De-gear when the average absolute combined forecast is weak.

    The filter is causal: the historical bottom-quartile benchmark is computed
    from a lagged rolling window, and today's mean absolute forecast is compared
    to it. If it is in the weak tail, positions are scaled to `scale`.

    This targets low-trend regimes like 2023-26, but the threshold is easy to
    overfit to recent data — keep opt-in and validate on pre-2023 data only.
    """
    if forecasts.empty:
        return pd.Series(1.0, index=forecasts.index)
    mean_abs = forecasts.abs().mean(axis=1)
    # Lagged rolling quantile of mean absolute forecast.
    bench = mean_abs.shift(1).rolling(window=window, min_periods=max(20, window // 3)).quantile(threshold)
    mult = pd.Series(scale, index=forecasts.index)
    mult[mean_abs >= bench] = 1.0
    return mult.fillna(1.0)


def corr_spike_overlay(
    returns: pd.DataFrame,
    span: int = 60,
    threshold: float = 0.5,
    max_degross: float = 0.5,
) -> pd.Series:
    """Portfolio-level de-grossing when average pairwise correlation spikes.

    Uses a lagged rolling average pairwise correlation (no lookahead). When the
    correlation is at or below `threshold` the multiplier is 1.0; as correlation
    rises toward 1.0 the multiplier falls linearly to `max_degross`.
    """
    rets = returns.dropna(how="all")
    if rets.shape[1] < 2:
        return pd.Series(1.0, index=rets.index)

    min_periods = max(2, span // 2)
    # Rolling correlation using only information up to t-1.
    rolled = rets.shift(1).rolling(window=span, min_periods=min_periods).corr()

    def _avg_offdiag(group: pd.DataFrame) -> float:
        n = group.shape[0]
        if n < 2:
            return float("nan")
        vals = group.values
        total = vals.sum()
        trace = np.trace(vals)
        return (total - trace) / (n * (n - 1))

    mean_corr = rolled.groupby(level=0).apply(_avg_offdiag)
    if threshold >= 1.0:
        mult = pd.Series(1.0, index=mean_corr.index)
    else:
        mult = 1.0 - (mean_corr - threshold) / (1.0 - threshold) * (1.0 - max_degross)
    mult = mult.clip(max_degross, 1.0)
    return mult.reindex(returns.index).fillna(1.0)


def drawdown_overlay(
    gross_returns: pd.Series,
    threshold: float = 0.10,
    scale: float = 0.50,
    recovery: float = 0.05,
) -> pd.Series:
    """Scale positions down during realised drawdowns and back up on recovery.

    The overlay is causal: the multiplier applied on day t uses the drawdown
    computed from gross returns only through day t-1. When drawdown (from peak)
    exceeds `threshold`, exposure is scaled to `scale`. It stays at `scale` until
    drawdown recovers to `recovery` or below, then returns to 1.0.

    Parameters mirror the research finding that trend strategies tend to recover
    after deep drawdowns, but the threshold is easy to overfit — keep opt-in.
    """
    gr = gross_returns.dropna()
    if gr.empty:
        return pd.Series(1.0, index=gross_returns.index)
    # Cumulative equity and running peak using returns through t-1.
    equity = (1.0 + gr).cumprod()
    peak = equity.shift(1).expanding(min_periods=2).max()
    dd = (equity - peak) / peak  # negative series
    mult = pd.Series(1.0, index=gr.index)
    triggered = False
    for i, (date, value) in enumerate(dd.items()):
        if i == 0:
            continue
        if not triggered and value <= -threshold:
            triggered = True
        elif triggered and value >= -recovery:
            triggered = False
        mult.iloc[i] = scale if triggered else 1.0
    return mult.reindex(gross_returns.index).fillna(1.0)
