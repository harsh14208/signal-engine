import numpy as np
import pandas as pd

from signal_engine.config import Config
from signal_engine.diagnostics import (
    cost_buffer_frontier,
    per_instrument_attribution,
    vix_regime_split,
)


def test_cost_buffer_frontier_shape(small_prices):
    cfg = Config()
    frontier = cost_buffer_frontier(
        small_prices, cfg, cost_values=(1.0, 2.0), buffer_values=(0.0, 0.1)
    )
    assert frontier.shape == (4, 7)
    assert "net_sharpe" in frontier.columns
    assert "turnover" in frontier.columns


def test_per_instrument_attribution_sums(result):
    attr = per_instrument_attribution(result)
    # Net contributions roughly sum to the portfolio mean daily return.
    assert "net_contrib" in attr.columns
    assert "notional_share" in attr.columns
    total = attr["net_contrib"].sum()
    port_mean = result.daily_returns.mean()
    assert abs(total - port_mean) < 1e-6


def test_vix_regime_split():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2015-01-01", periods=400)
    rets = pd.Series(rng.normal(0.0003, 0.01, size=400), index=idx)
    vix = pd.Series(rng.uniform(10, 40, size=400), index=idx)
    split = vix_regime_split(rets, vix)
    assert "high_vix" in split and "low_vix" in split
    assert split["high_vix"]["n_days"] + split["low_vix"]["n_days"] == 400
