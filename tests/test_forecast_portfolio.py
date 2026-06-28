import numpy as np
import pandas as pd

from signal_engine.config import FDM_CAP, FORECAST_CAP, IDM_CAP
from signal_engine.forecast import combine_instrument, equal_weights, estimate_fdm
from signal_engine.portfolio import apply_buffer, estimate_idm, position_units, vol_governor


def test_equal_weights_sum_to_one():
    w = equal_weights(["a", "b", "c", "d"])
    assert abs(sum(w.values()) - 1.0) < 1e-12


def test_fdm_within_bounds():
    corr = pd.DataFrame(
        [[1.0, 0.2, 0.1], [0.2, 1.0, 0.15], [0.1, 0.15, 1.0]],
        index=["x", "y", "z"],
        columns=["x", "y", "z"],
    )
    fdm = estimate_fdm(corr, equal_weights(["x", "y", "z"]))
    assert 1.0 <= fdm <= FDM_CAP + 1e-9


def test_combine_capped():
    idx = pd.RangeIndex(100)
    fc = {"a": pd.Series(15.0, idx), "b": pd.Series(15.0, idx)}
    out = combine_instrument(fc, fdm=2.5)
    assert out.abs().max() <= FORECAST_CAP + 1e-9


def test_combine_accepts_dynamic_fdm():
    idx = pd.RangeIndex(100)
    fc = {"a": pd.Series(10.0, idx), "b": pd.Series(0.0, idx)}
    fdm = pd.Series(1.5, index=idx)
    fdm.iloc[:50] = 1.0
    out = combine_instrument(fc, fdm=fdm)
    assert out.abs().max() <= FORECAST_CAP + 1e-9
    # Equal rule weights → base combined forecast = 5.0; scaled by fdm.
    assert (out.iloc[:50] == 5.0).all()
    assert (out.iloc[50:] == 7.5).all()


def test_idm_within_bounds():
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.normal(0, 0.01, size=(500, 4)), columns=list("abcd"))
    idm = estimate_idm(rets, equal_weights(list("abcd")))
    assert 1.0 <= idm <= IDM_CAP + 1e-9


def test_position_sign_and_zero():
    idx = pd.date_range("2015-01-01", periods=50)
    price = pd.Series(100.0, idx)
    vol = pd.Series(0.15, idx)
    long = position_units(pd.Series(10.0, idx), price, vol, 1e6, 0.2, 0.1, 2.0)
    flat = position_units(pd.Series(0.0, idx), price, vol, 1e6, 0.2, 0.1, 2.0)
    assert (long > 0).all()
    assert (flat == 0).all()


def test_position_units_accepts_dynamic_inputs():
    idx = pd.date_range("2015-01-01", periods=50)
    price = pd.Series(100.0, idx)
    vol = pd.Series(0.15, idx)
    forecast = pd.Series(10.0, idx)
    weight = pd.Series(0.1, idx)
    weight.iloc[25:] = 0.2
    idm = pd.Series(2.0, idx)
    out = position_units(forecast, price, vol, 1e6, 0.2, weight, idm)
    # Higher weight after day 25 → larger positions.
    assert out.iloc[30:].mean() > out.iloc[:20].mean()


def test_buffer_reduces_turnover():
    rng = np.random.default_rng(1)
    noisy = pd.Series(rng.normal(0, 100, size=2000)).cumsum() / 50
    raw_changes = (noisy.diff() != 0).sum()
    buffered = apply_buffer(noisy, fraction=0.10)
    buf_changes = (buffered.diff().fillna(0) != 0).sum()
    assert buf_changes < raw_changes


def test_buffer_no_lookahead():
    """The band before min_periods must not be influenced by later values."""
    rng = np.random.default_rng(2)
    base = pd.Series(rng.normal(0, 1.0, size=60)).cumsum()
    a = base.copy()
    b = base.copy()
    b.iloc[30:] += 1_000_0  # huge future values
    buf_a = apply_buffer(a, fraction=0.10)
    buf_b = apply_buffer(b, fraction=0.10)
    # Through the first 20 bars the band uses only data seen so far.
    assert np.allclose(buf_a.iloc[:20].to_numpy(), buf_b.iloc[:20].to_numpy())


def test_vol_governor_no_lookahead():
    """A shock after day t must not change governor values at or before t."""
    rng = np.random.default_rng(3)
    base = pd.Series(rng.normal(0, 0.01, size=100))
    g1 = vol_governor(base, target_vol=0.20)
    shocked = base.copy()
    shocked.iloc[50:] *= 10.0
    g2 = vol_governor(shocked, target_vol=0.20)
    assert np.allclose(g1.iloc[:50].to_numpy(), g2.iloc[:50].to_numpy())
