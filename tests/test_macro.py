import numpy as np
import pandas as pd
import pytest

from signal_engine.macro import (
    credit_overlay,
    hmm_regime_overlay,
    regime_overlay,
    vix_term_overlay,
)


def test_regime_overlay_low_vix_is_one():
    idx = pd.bdate_range("2015-01-01", periods=200)
    prices = pd.DataFrame({"SPY": 100 + np.arange(200) * 0.1}, index=idx)
    vix = pd.Series(15.0, index=idx)
    mult = regime_overlay(prices, vix, vix_threshold=20.0, max_degear=0.5)
    assert (mult == 1.0).all()


def test_regime_overlay_high_vix_reduces_multiplier():
    idx = pd.bdate_range("2015-01-01", periods=200)
    prices = pd.DataFrame({"SPY": 100 + np.arange(200) * 0.1}, index=idx)
    vix = pd.Series(35.0, index=idx)
    mult = regime_overlay(prices, vix, vix_threshold=20.0, vix_cap=40.0, max_degear=0.5)
    # After the lag, the multiplier should be below 1.0.
    assert mult.iloc[2:].mean() < 0.9
    assert mult.min() <= 0.75


def test_regime_overlay_drawdown_reduces_multiplier():
    idx = pd.bdate_range("2015-01-01", periods=200)
    prices = pd.DataFrame(
        {"SPY": np.concatenate([np.linspace(100, 130, 100), np.linspace(130, 100, 100)])}, index=idx
    )
    vix = pd.Series(15.0, index=idx)
    mult = regime_overlay(prices, vix, drawdown_threshold=-0.05, max_degear=0.5)
    assert mult.min() < 1.0


def test_regime_overlay_no_lookahead():
    idx = pd.bdate_range("2015-01-01", periods=150)
    prices = pd.DataFrame({"SPY": 100 + np.arange(150) * 0.1}, index=idx)
    vix = pd.Series(15.0, index=idx)
    base = regime_overlay(prices, vix)
    tweaked_vix = vix.copy()
    tweaked_vix.iloc[-1] = 50.0
    mod = regime_overlay(prices, tweaked_vix)
    assert np.allclose(base.iloc[:-1].to_numpy(), mod.iloc[:-1].to_numpy())


def test_vix_term_overlay_stress_degears_and_calm_gears():
    idx = pd.bdate_range("2015-01-01", periods=50)
    df = pd.DataFrame({"vix": 20.0, "vix9d": 20.0, "vix3m": 20.0}, index=idx)

    # Short-end fear spike.
    stress = df.copy()
    stress.loc[idx[10:], "vix9d"] = 30.0
    mult_stress = vix_term_overlay(stress, short_thresh=1.10, long_thresh=0.95)
    assert mult_stress.iloc[11:].mean() < 1.0
    assert mult_stress.min() <= 0.5

    # Calm term structure (3M well below spot).
    calm = df.copy()
    calm.loc[idx[10:], "vix3m"] = 15.0
    mult_calm = vix_term_overlay(calm, short_thresh=1.10, long_thresh=0.95)
    assert mult_calm.iloc[11:].mean() > 1.0


def test_vix_term_overlay_no_lookahead():
    idx = pd.bdate_range("2015-01-01", periods=50)
    df = pd.DataFrame({"vix": 20.0, "vix9d": 20.0, "vix3m": 20.0}, index=idx)
    base = vix_term_overlay(df)
    tweaked = df.copy()
    tweaked.loc[idx[-1], "vix9d"] = 50.0
    mod = vix_term_overlay(tweaked)
    assert np.allclose(base.iloc[:-1].to_numpy(), mod.iloc[:-1].to_numpy())


def test_credit_overlay_stress_degears_and_calm_gears():
    idx = pd.bdate_range("2015-01-01", periods=500)
    base = pd.Series(2.0, index=idx)

    # Spike the spread far above its trailing median → de-gear.
    high = base.copy()
    high.iloc[200:230] = 10.0
    mult_high = credit_overlay(high, upper_thresh=1.5, lower_thresh=0.8, lookback=60)
    assert mult_high.iloc[210:230].mean() < 1.0
    assert mult_high.min() <= 0.5

    # Crash the spread far below its trailing median → gear up.
    low = base.copy()
    low.iloc[200:230] = 0.5
    mult_low = credit_overlay(low, upper_thresh=1.5, lower_thresh=0.8, lookback=60)
    assert mult_low.iloc[210:230].mean() > 1.0


def test_credit_overlay_no_lookahead():
    idx = pd.bdate_range("2015-01-01", periods=200)
    spread = pd.Series(2.0, index=idx)
    base = credit_overlay(spread)
    tweaked = spread.copy()
    tweaked.iloc[-1] = 10.0
    mod = credit_overlay(tweaked)
    assert np.allclose(base.iloc[:-1].to_numpy(), mod.iloc[:-1].to_numpy())


def test_hmm_regime_overlay_runs_and_no_lookahead():
    pytest.importorskip("hmmlearn")
    idx = pd.bdate_range("2015-01-01", periods=400)
    np.random.seed(0)
    prices = pd.DataFrame(
        {"SPY": 100 * (1 + np.random.normal(0.0003, 0.01, len(idx))).cumprod()}, index=idx
    )
    vix = pd.Series(15 + np.random.normal(0, 2, len(idx)).clip(0), index=idx)
    spy = prices["SPY"]
    tnx = pd.Series(0.02, index=idx)
    irx = pd.Series(0.005, index=idx)

    base = hmm_regime_overlay(prices, vix, spy, tnx=tnx, irx=irx, train_window=200, refit_stride=21)
    assert len(base) == len(idx)
    assert base.notna().any()
    assert base.min() >= 0.5
    assert base.max() <= 1.2

    tweaked_vix = vix.copy()
    tweaked_vix.iloc[-1] = 80.0
    mod = hmm_regime_overlay(
        prices, tweaked_vix, spy, tnx=tnx, irx=irx, train_window=200, refit_stride=21
    )
    assert np.allclose(base.iloc[:-1].to_numpy(), mod.iloc[:-1].to_numpy())
