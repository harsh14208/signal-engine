import numpy as np
import pandas as pd

from signal_engine.backtest import run_backtest
from signal_engine.config import Config
from signal_engine.data import synthetic_prices
from signal_engine.markets import symbols
from signal_engine.portfolio import vol_governor


def test_governor_clamped_and_finite():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 500))
    g = vol_governor(r, 0.20, lo=0.20, hi=2.50)
    assert g.notna().all()
    assert (g >= 0.20 - 1e-9).all() and (g <= 2.50 + 1e-9).all()


def test_governor_inverse_to_vol():
    # A high-vol stream is scaled DOWN; a low-vol stream UP (until clamped).
    hi = pd.Series(np.random.default_rng(1).normal(0, 0.030, 800))  # ~48% ann
    lo = pd.Series(np.random.default_rng(2).normal(0, 0.004, 800))  # ~6% ann
    assert vol_governor(hi, 0.20).iloc[-1] < vol_governor(lo, 0.20).iloc[-1]


def test_governor_no_lookahead():
    # The multiplier is lagged: replacing the LAST return must not change any
    # governor value except possibly the final one.
    base = pd.Series(np.random.default_rng(3).normal(0, 0.01, 400))
    g1 = vol_governor(base, 0.20)
    tweaked = base.copy()
    tweaked.iloc[-1] = 0.5  # huge shock on the last day
    g2 = vol_governor(tweaked, 0.20)
    assert np.allclose(g1.iloc[:-1].to_numpy(), g2.iloc[:-1].to_numpy())


def test_governor_improves_vol_targeting():
    px = synthetic_prices(symbols(), n_days=1500, seed=9)
    on = run_backtest(px, Config(use_governor=True)).daily_returns.std() * 16
    off = run_backtest(px, Config(use_governor=False)).daily_returns.std() * 16
    target = 0.20
    # Governed realised vol must be no further from target than ungoverned, and land
    # in a sane band around it.
    assert abs(on - target) <= abs(off - target) + 0.02
    assert 0.12 < on < 0.30


def test_governor_smoothing_reduces_daily_changes():
    r = pd.Series(np.random.default_rng(4).normal(0, 0.01, 800))
    raw = vol_governor(r, 0.20, smooth=None)
    smooth = vol_governor(r, 0.20, smooth=8)
    raw_changes = raw.diff().abs().sum()
    smooth_changes = smooth.diff().abs().sum()
    assert smooth_changes < raw_changes
    assert (smooth >= 0.20 - 1e-9).all() and (smooth <= 2.50 + 1e-9).all()
