import numpy as np

from signal_engine.data import random_walk_panel, synthetic_carry, synthetic_prices
from signal_engine.markets import symbols


def test_synthetic_prices_shape_and_positive():
    syms = symbols()[:5]
    px = synthetic_prices(syms, n_days=500, seed=1)
    assert list(px.columns) == syms
    assert len(px) == 500
    assert (px > 0).all().all()


def test_synthetic_prices_deterministic():
    a = synthetic_prices(["SPY", "GLD"], n_days=300, seed=42)
    b = synthetic_prices(["SPY", "GLD"], n_days=300, seed=42)
    assert np.allclose(a.values, b.values)


def test_random_walk_has_near_zero_drift():
    px = random_walk_panel(8, 4000, seed=2)
    ann_drift = px.pct_change().mean() * 256
    # Driftless by construction → mean annual return hugs zero.
    assert ann_drift.abs().mean() < 0.08


def test_synthetic_carry_shape():
    px = synthetic_prices(["TLT", "IEF"], n_days=200, seed=1)
    c = synthetic_carry(["TLT", "IEF"], px.index)
    assert c.shape == (200, 2)
