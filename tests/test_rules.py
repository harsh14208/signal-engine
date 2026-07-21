import numpy as np
import pandas as pd

from signal_engine.config import DEFAULT_EWMAC_SPEEDS, FORECAST_CAP
from signal_engine.rules import (
    acceleration_forecast,
    breakout_forecast,
    carry_forecast,
    cross_sectional_momentum_forecast,
    ewmac_forecast,
    trend_forecasts,
)
from signal_engine.volatility import blended_daily_vol, daily_returns


def test_ewmac_forecast_capped(small_prices):
    r = daily_returns(small_prices["SPY"])
    v = blended_daily_vol(r)
    f = ewmac_forecast(small_prices["SPY"], v, 16, 64)
    assert f.dropna().abs().max() <= FORECAST_CAP + 1e-9
    assert f.iloc[200:].notna().any()


def test_ewmac_sign_follows_trend():
    # Monotonically rising price → positive (long) trend forecast.
    px = pd.Series(100 * (1.001 ** np.arange(600)))
    v = blended_daily_vol(px.pct_change())
    f = ewmac_forecast(px, v, 16, 64)
    assert f.iloc[-1] > 0


def test_breakout_capped(small_prices):
    f = breakout_forecast(small_prices["TLT"], 40)
    assert f.dropna().abs().max() <= FORECAST_CAP + 1e-9


def test_carry_sign():
    idx = pd.date_range("2015-01-01", periods=300)
    pos = carry_forecast(pd.Series(0.05, idx), pd.Series(0.15, idx))
    neg = carry_forecast(pd.Series(-0.05, idx), pd.Series(0.15, idx))
    assert (pos > 0).all() and (neg < 0).all()


def test_trend_forecasts_keys(small_prices):
    r = daily_returns(small_prices["SPY"])
    v = blended_daily_vol(r)
    out = trend_forecasts(small_prices["SPY"], v, DEFAULT_EWMAC_SPEEDS, (40, 80))
    assert "ewmac_16_64" in out and "breakout_40" in out
    assert len(out) == len(DEFAULT_EWMAC_SPEEDS) + 2


def test_acceleration_forecast_capped(small_prices):
    r = daily_returns(small_prices["SPY"])
    v = blended_daily_vol(r)
    f = acceleration_forecast(small_prices["SPY"], v)
    assert f.dropna().abs().max() <= FORECAST_CAP + 1e-9


def test_acceleration_positive_in_accelerating_uptrend():
    # Price path with increasing slope plus a little noise → positive acceleration
    # at some point during the acceleration phase.
    rng = np.random.default_rng(7)
    t = np.arange(300)
    px = pd.Series(100 * (1 + 0.0001 * t + 0.000005 * t**2) + rng.normal(0, 0.05, 300))
    v = blended_daily_vol(px.pct_change())
    f = acceleration_forecast(px, v)
    assert f.dropna().max() > 0


def test_cross_sectional_momentum_capped(small_prices):
    f = cross_sectional_momentum_forecast(small_prices, lookback=40)
    assert (f.abs().fillna(0) <= FORECAST_CAP + 1e-9).all().all()


def test_cross_sectional_momentum_ranking(small_prices):
    f = cross_sectional_momentum_forecast(small_prices, lookback=40)
    # For each row, the instrument with the highest recent return should have
    # the highest forecast and the lowest the lowest.
    mom = small_prices.pct_change(40)
    for idx in f.index[50:]:
        row = f.loc[idx].dropna()
        if len(row) < 2:
            continue
        best = row.idxmax()
        worst = row.idxmin()
        assert mom.loc[idx, best] >= mom.loc[idx, worst]


def test_custom_ewmac_speed_no_lookahead(small_prices):
    """Custom EWMAC speeds not in the published table must use an expanding scalar."""
    r = daily_returns(small_prices["SPY"])
    v = blended_daily_vol(r)
    base = ewmac_forecast(small_prices["SPY"], v, 10, 50)
    tweaked = small_prices["SPY"].copy()
    tweaked.iloc[-1] *= 1.5
    rt = daily_returns(tweaked)
    vt = blended_daily_vol(rt)
    mod = ewmac_forecast(tweaked, vt, 10, 50)
    # Earlier forecasts should be unchanged by a perturbation at the end.
    assert np.allclose(base.iloc[:-1], mod.iloc[:-1], equal_nan=True)


def test_custom_breakout_span_no_lookahead(small_prices):
    """Custom breakout spans not in the published table must use an expanding scalar."""
    base = breakout_forecast(small_prices["TLT"], 25)
    tweaked = small_prices["TLT"].copy()
    tweaked.iloc[-1] *= 1.5
    mod = breakout_forecast(tweaked, 25)
    assert np.allclose(base.iloc[:-1], mod.iloc[:-1], equal_nan=True)
