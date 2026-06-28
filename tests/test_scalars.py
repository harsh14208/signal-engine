import numpy as np
import pandas as pd

from signal_engine.scalars import _dynamic_scalar


def test_dynamic_scalar_targets_mean_abs():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2015-01-01", periods=500)
    # A forecast whose mean abs is ~5.
    fc = pd.Series(rng.normal(0, 6.25, size=500), index=idx)
    scalar = _dynamic_scalar(fc, target=10.0)
    scaled = (fc * scalar).iloc[100:]  # skip initial warm-up
    mean_abs = scaled.abs().mean()
    assert 9.0 < mean_abs < 11.0


def test_dynamic_scalar_no_lookahead():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2015-01-01", periods=200)
    fc = pd.Series(rng.normal(0, 5, size=200), index=idx)
    base = _dynamic_scalar(fc)
    tweaked = fc.copy()
    tweaked.iloc[-1] = 100.0
    mod = _dynamic_scalar(tweaked)
    assert np.allclose(base.iloc[:-1].to_numpy(), mod.iloc[:-1].to_numpy())
