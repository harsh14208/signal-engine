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

import numpy as np
import pandas as pd

from .config import ANNUAL_VOL_SQRT, BUSINESS_DAYS_YEAR


def rolling_sharpe(daily: pd.Series, window: int = BUSINESS_DAYS_YEAR) -> pd.Series:
    """Annualised Sharpe over a trailing `window` (default ~1 year)."""
    d = daily.dropna()
    mean = d.rolling(window).mean()
    std = d.rolling(window).std()
    return (mean / std * ANNUAL_VOL_SQRT).rename("rolling_sharpe")


def edge_decay_report(
    daily: pd.Series,
    window: int = BUSINESS_DAYS_YEAR,
    alarm_floor: float = 0.0,
    alarm_on_worst_quartile: bool = False,
) -> dict:
    """Summarise the rolling-Sharpe path and raise an alarm if the latest dips
    below `alarm_floor` — a cheap, honest 'is the edge still alive' monitor.

    Two alarm modes:
      • `alarm_floor` — the crude absolute floor (default: rolling Sharpe < 0).
      • `worst_quartile` — the latest rolling Sharpe sits in the worst quartile of
        its own history. This self-calibrating flag (arXiv 2604.18821) catches a
        *decaying* edge earlier than the absolute floor, and adapts to a strategy
        whose healthy Sharpe was never far above zero to begin with. Enable it by
        passing `alarm_on_worst_quartile=True`.
    """
    rs = rolling_sharpe(daily, window).dropna()
    if rs.empty:
        return {"insufficient": True}
    current = float(rs.iloc[-1])
    floor_alarm = current < alarm_floor
    # Quartile flag needs enough rolling points to be meaningful; gate it so it
    # doesn't fire spuriously on a handful of early windows.
    q25 = float(rs.quantile(0.25)) if len(rs) >= 60 else None
    worst_quartile = bool(q25 is not None and current <= q25)
    alarm = bool(floor_alarm or (alarm_on_worst_quartile and worst_quartile))
    return {
        "window": window,
        "current": current,
        "median": float(rs.median()),
        "min": float(rs.min()),
        "max": float(rs.max()),
        "q25": q25,
        "pct_windows_below_zero": float((rs < 0).mean()),
        "alarm_floor": alarm_floor,
        "alarm_on_worst_quartile": alarm_on_worst_quartile,
        "alarm": alarm,
        "worst_quartile": worst_quartile,
        "decay_warning": bool(floor_alarm or worst_quartile),
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
    out = {
        "n": int(len(df)),
        "corr": corr,
        "tracking_error": tracking_error,
        "drift": drift,
        "aligned": corr > 0.80 and tracking_error < 0.05,
    }
    out["drift_decomposition"] = decompose_drift(df["live"], df["bt"])
    return out


def decompose_drift(live_returns: pd.Series, backtest_returns: pd.Series) -> dict:
    """Attribute the live-vs-model drift into execution-quality components.

    A single `drift` number conflates several causes. Following the spirit of
    implementation-shortfall analysis (Perold), we regress live on modeled returns
    (live = α + β·model + ε) and split the annualised drift into:

      • `alpha`      — return the live book earns that the model does *not* explain;
                       persistent, sign-stable slippage (real cost, data lag).
      • `beta_gap`   — drift from running a *different exposure* than modeled
                       (β ≠ 1): (β − 1)·mean(model), i.e. under/over-replication.
      • `residual`   — annualised idiosyncratic tracking noise (std of ε).

    alpha + beta_gap reconstruct the mean drift; residual sizes the noise around
    it. A large `beta_gap` says "fix your sizing/replication"; a large negative
    `alpha` says "your real costs exceed the assumed model cost."
    """
    df = pd.concat([live_returns.rename("live"), backtest_returns.rename("bt")], axis=1).dropna()
    if len(df) < 20 or df["bt"].std() == 0:
        return {"insufficient": True, "n": int(len(df))}
    x = df["bt"].to_numpy()
    y = df["live"].to_numpy()
    beta, alpha = np.polyfit(x, y, 1)
    resid = y - (alpha + beta * x)
    mean_model = float(df["bt"].mean())
    alpha_ann = float(alpha * BUSINESS_DAYS_YEAR)
    beta_gap_ann = float((beta - 1.0) * mean_model * BUSINESS_DAYS_YEAR)
    return {
        "n": int(len(df)),
        "beta": float(beta),
        "alpha": alpha_ann,
        "beta_gap": beta_gap_ann,
        "residual": float(resid.std() * ANNUAL_VOL_SQRT),
        "total_drift": float(alpha_ann + beta_gap_ann),
    }


def implementation_shortfall_components(
    live_returns: pd.Series,
    backtest_returns: pd.Series,
    decision_prices: pd.Series | None = None,
    arrival_prices: pd.Series | None = None,
) -> dict:
    """Perold-style implementation-shortfall decomposition when price data is available.

    Components (all annualised where applicable):
      • `delay`      — decision close → arrival price (overnight drift before fill).
      • `opportunity` — arrival price → next close (strategy return after entry).
      • `execution`  — residual cost not explained by delay/opportunity (spread + impact).
      • `regression` — the regression-based alpha/beta/residual decomposition above.

    If `decision_prices` or `arrival_prices` are absent, only the regression
    decomposition is returned.
    """
    out = {"regression": decompose_drift(live_returns, backtest_returns)}
    if decision_prices is None or arrival_prices is None:
        return out

    shared = live_returns.index.intersection(decision_prices.index).intersection(arrival_prices.index)
    if len(shared) < 20:
        out["insufficient_prices"] = True
        return out

    decision = decision_prices.reindex(shared)
    arrival = arrival_prices.reindex(shared)
    # Delay: overnight return from decision close to arrival price.
    delay_daily = (arrival - decision) / decision
    # Opportunity: live return minus delay (what the strategy made after entry).
    opportunity_daily = live_returns.reindex(shared) - delay_daily
    out["delay"] = float(delay_daily.mean() * BUSINESS_DAYS_YEAR)
    out["opportunity"] = float(opportunity_daily.mean() * BUSINESS_DAYS_YEAR)
    # Execution residual: what is left of the live-vs-model drift after delay.
    diff = (live_returns - backtest_returns).reindex(shared)
    out["execution_residual"] = float(
        (diff.mean() - delay_daily.mean()) * BUSINESS_DAYS_YEAR
    )
    return out
