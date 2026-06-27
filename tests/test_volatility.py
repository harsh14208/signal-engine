import numpy as np

from signal_engine.volatility import annualise, blended_daily_vol, daily_returns


def test_blended_vol_positive_after_warmup(small_prices):
    r = daily_returns(small_prices["SPY"])
    v = blended_daily_vol(r)
    tail = v.iloc[100:]
    assert (tail > 0).all()
    assert not tail.isna().any()


def test_annualise_scales_by_sqrt_256(small_prices):
    r = daily_returns(small_prices["SPY"])
    v = blended_daily_vol(r)
    assert np.allclose((annualise(v) / v).dropna(), 16.0)


def test_blended_vol_tracks_magnitude():
    import pandas as pd

    calm = pd.Series(np.random.default_rng(0).normal(0, 0.005, 1000))
    wild = pd.Series(np.random.default_rng(0).normal(0, 0.02, 1000))
    assert blended_daily_vol(wild).iloc[-1] > blended_daily_vol(calm).iloc[-1]
