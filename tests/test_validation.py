import numpy as np
import pandas as pd

from signal_engine.backtest import run_backtest
from signal_engine.config import Config
from signal_engine.validation import (
    block_bootstrap_sharpe,
    lo_sharpe_ci,
    placebo_sharpes,
    random_walk_panel,
)


def _series(sharpe_target=0.05, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    # daily series with a small positive mean → modest positive Sharpe
    return pd.Series(rng.normal(sharpe_target * 0.01, 0.01, n))


def test_lo_ci_brackets_sharpe():
    d = _series()
    out = lo_sharpe_ci(d, n_trials=50)
    assert out["ci_low"] <= out["sharpe"] <= out["ci_high"]
    assert "passes_deflated" in out and "zero_inside" in out


def test_block_bootstrap_ordering():
    d = _series()
    out = block_bootstrap_sharpe(d, n_sims=300)
    assert out["p5"] <= out["p50"] <= out["p95"]


def test_random_walk_panel_shape():
    px = random_walk_panel(6, 800, seed=1)
    assert px.shape == (800, 6)
    assert (px > 0).all().all()


def test_placebo_floor_near_zero():
    # The strategy on driftless data should produce a noise floor near zero.
    cfg = Config()
    out = placebo_sharpes(
        lambda panel: run_backtest(panel, cfg).daily_returns,
        n_placebo=5,
        n_instruments=6,
        n_days=900,
    )
    assert out["n_placebo"] >= 1
    assert abs(out["mean"]) < 0.6  # centered near zero, not a structural edge
