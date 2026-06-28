import numpy as np
import pandas as pd

from signal_engine.backtest import run_backtest
from signal_engine.config import Config
from signal_engine.validation import (
    block_bootstrap_sharpe,
    lo_sharpe_ci,
    placebo_sharpes,
    purged_walk_forward,
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


def test_walk_forward_folds_are_chronological(small_prices):
    cfg = Config()
    out = purged_walk_forward(small_prices, cfg, n_splits=3)
    assert not out.get("insufficient")
    assert out["n_folds"] >= 1
    for f in out["folds"]:
        assert f["test_start"] > f["train_end"]


def test_walk_forward_embargo_creates_gap(small_prices):
    cfg = Config()
    out = purged_walk_forward(small_prices, cfg, n_splits=4, embargo_frac=0.02)
    assert not out.get("insufficient")
    # The test window should start strictly after the train end.
    for f in out["folds"]:
        train_end_idx = small_prices.index.get_loc(pd.Timestamp(f["train_end"]))
        test_start_idx = small_prices.index.get_loc(pd.Timestamp(f["test_start"]))
        assert test_start_idx > train_end_idx
