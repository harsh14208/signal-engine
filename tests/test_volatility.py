import numpy as np
import pytest

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


def test_garch_blend_fallback_when_arch_missing(monkeypatch, small_prices):
    """If arch is not installed, GARCH request silently falls back to EWMA."""
    import signal_engine.volatility as vol_mod

    monkeypatch.setattr(vol_mod, "_GARCH_AVAILABLE", False)
    r = daily_returns(small_prices["SPY"])
    v = blended_daily_vol(r, use_garch=True, garch_weight=1.0)
    tail = v.iloc[100:]
    assert (tail > 0).all()
    assert not tail.isna().any()


def test_garch_blend_positive_when_available(small_prices):
    pytest.importorskip("arch")
    r = daily_returns(small_prices["SPY"])
    v = blended_daily_vol(r, use_garch=True, garch_weight=1.0)
    tail = v.iloc[400:]
    assert (tail > 0).all()
    assert not tail.isna().any()


def test_garch_no_lookahead(small_prices):
    pytest.importorskip("arch")
    r = daily_returns(small_prices["SPY"])
    base = blended_daily_vol(r, use_garch=True, garch_weight=1.0)
    tweaked = r.copy()
    tweaked.iloc[-1] = tweaked.iloc[-1] + 0.5
    mod = blended_daily_vol(tweaked, use_garch=True, garch_weight=1.0)
    assert np.allclose(
        base.iloc[:-1].to_numpy(), mod.iloc[:-1].to_numpy(), equal_nan=True
    )
