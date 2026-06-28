"""Research diagnostics — cost/buffer frontier, attribution, and regime splits."""

from __future__ import annotations

import pandas as pd

from .backtest import BacktestResult, run_backtest
from .config import Config
from .metrics import sharpe


def cost_buffer_frontier(
    prices: pd.DataFrame,
    config: Config,
    cost_values: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0),
    buffer_values: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20),
) -> pd.DataFrame:
    """Grid-search over cost and buffer values, returning a comparison table."""
    rows = []
    n = len(prices)
    split = int(n * 0.7)
    for cost in cost_values:
        for buf in buffer_values:
            cfg = Config(
                cost_bps=cost,
                buffer_fraction=buf,
                use_governor=config.use_governor,
                weight_scheme=config.weight_scheme,
                use_carry=config.use_carry,
                use_accel=config.use_accel,
                use_xsmom=config.use_xsmom,
                use_corr_spike=config.use_corr_spike,
                use_regime_overlay=config.use_regime_overlay,
            )
            result = run_backtest(prices, cfg)
            daily = result.daily_returns
            is_sr = sharpe(daily.iloc[:split])
            oos_sr = sharpe(daily.iloc[split:])
            rows.append(
                {
                    "cost_bps": cost,
                    "buffer": buf,
                    "net_sharpe": sharpe(daily),
                    "is_sharpe": is_sr,
                    "oos_sharpe": oos_sr,
                    "turnover": float(result.turnover.mean() * 256),
                    "max_dd": float(
                        (result.equity / result.equity.cummax() - 1.0).min()
                    ),
                }
            )
    return pd.DataFrame(rows)


def per_instrument_attribution(result: BacktestResult) -> pd.DataFrame:
    """Mean daily gross P&L, cost, and net contribution per instrument."""
    gross = result.per_instrument_gross.mean()
    net = result.per_instrument_returns.mean()
    df = pd.DataFrame(
        {
            "gross_contrib": gross,
            "cost": gross - net,
            "net_contrib": net,
            "notional_share": result.notional.abs().mean() / result.notional.abs().mean().sum(),
        }
    )
    df = df.sort_values("net_contrib", ascending=False)
    df["cum_net"] = df["net_contrib"].cumsum()
    return df


def vix_regime_split(daily_returns: pd.Series, vix: pd.Series) -> dict:
    """Split performance by high vs low VIX to diagnose regime dependence."""
    aligned_vix = vix.reindex(daily_returns.index).ffill()
    median_vix = aligned_vix.median()
    high = daily_returns[aligned_vix > median_vix]
    low = daily_returns[aligned_vix <= median_vix]

    def _stats(s: pd.Series) -> dict:
        s = s.dropna()
        eq = (1.0 + s).cumprod()
        return {
            "n_days": len(s),
            "sharpe": sharpe(s),
            "ann_vol": float(s.std() * 16),
            "ann_return": float(s.mean() * 256),
            "max_dd": float((eq / eq.cummax() - 1.0).min()),
        }

    return {
        "median_vix": float(median_vix),
        "high_vix": _stats(high),
        "low_vix": _stats(low),
    }
