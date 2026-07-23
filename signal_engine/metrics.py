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


def buy_and_hold_summary(prices: pd.Series) -> dict:
    """Sharpe/CAGR/vol/MaxDD of simply buying and holding a single price series."""
    p = prices.dropna()
    if len(p) < 2:
        return {
            "sharpe": float("nan"), "cagr": float("nan"),
            "ann_vol": float("nan"), "max_drawdown": float("nan"),
        }
    daily = p.pct_change().dropna()
    equity = (1.0 + daily).cumprod()
    return {
        "sharpe": sharpe(daily), "cagr": cagr(equity),
        "ann_vol": ann_vol(daily), "max_drawdown": max_drawdown(equity),
    }


def benchmark_comparison(
    equity: pd.Series,
    daily: pd.Series,
    benchmark_prices: pd.Series,
    *,
    cagr_tolerance: float = 0.02,
    min_drawdown_improvement: float = 0.10,
) -> dict:
    """Compare a strategy against a trivial buy-and-hold benchmark over the SAME
    aligned date range — the "does this beat doing nothing" check no other gate
    in this codebase asks. A real edge that fails this at realistic leverage/
    financing costs doesn't justify the platform, however good its Sharpe looks
    in isolation.

    The pass condition (`beats_priority`) encodes CAGR > MaxDD > Sharpe > Calmar
    as the optimization order, not Sharpe-first: it passes if the strategy beats
    the benchmark's CAGR outright, OR its CAGR is within `cagr_tolerance`
    (absolute, default 2 percentage points) of the benchmark's AND its max
    drawdown is at least `min_drawdown_improvement` (default 10%) smaller in
    magnitude — i.e. a strategy that nearly matches buy-and-hold CAGR while
    drawing down meaningfully less is a genuine win on this priority, even if
    its Sharpe or Calmar don't lead. Sharpe/Calmar remain reported but are not
    gating criteria here.
    """
    d = _clean(daily)
    if len(d) < 2:
        return {"insufficient": True}
    bench_prices = benchmark_prices.reindex(d.index).dropna()
    if len(bench_prices) < 2:
        return {"insufficient": True}
    bench = buy_and_hold_summary(bench_prices)
    strat_cagr = cagr(equity)
    strat_sharpe = sharpe(d)
    strat_maxdd = max_drawdown(equity)
    bench_maxdd = bench["max_drawdown"]

    beats_cagr = bool(strat_cagr > bench["cagr"])
    beats_sharpe = bool(strat_sharpe > bench["sharpe"])
    cagr_close = bool(strat_cagr >= bench["cagr"] - cagr_tolerance)
    maxdd_meaningfully_better = bool(
        not np.isnan(strat_maxdd)
        and not np.isnan(bench_maxdd)
        and bench_maxdd != 0
        and abs(strat_maxdd) <= abs(bench_maxdd) * (1.0 - min_drawdown_improvement)
    )
    beats_priority = beats_cagr or (cagr_close and maxdd_meaningfully_better)

    return {
        "insufficient": False,
        "strategy_cagr": strat_cagr,
        "strategy_sharpe": strat_sharpe,
        "strategy_max_drawdown": strat_maxdd,
        "benchmark_cagr": bench["cagr"],
        "benchmark_sharpe": bench["sharpe"],
        "benchmark_max_drawdown": bench_maxdd,
        "beats_cagr": beats_cagr,
        "beats_sharpe": beats_sharpe,
        "cagr_close": cagr_close,
        "maxdd_meaningfully_better": maxdd_meaningfully_better,
        "beats_priority": beats_priority,
        "beats_both": beats_cagr and beats_sharpe,
    }


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
