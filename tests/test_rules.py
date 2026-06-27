import numpy as np
import pandas as pd

from signal_engine.config import DEFAULT_EWMAC_SPEEDS, FORECAST_CAP
from signal_engine.rules import (
    breakout_forecast,
    carry_forecast,
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
