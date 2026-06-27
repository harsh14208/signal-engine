import numpy as np
import pandas as pd

from signal_engine import metrics


def test_sharpe_matches_manual():
    rng = np.random.default_rng(0)
    d = pd.Series(rng.normal(0.0004, 0.01, 5000))
    expected = d.mean() / d.std() * 16
    assert abs(metrics.sharpe(d) - expected) < 1e-9


def test_max_drawdown_known_curve():
    eq = pd.Series([1.0, 1.2, 0.9, 1.5])  # peak 1.2 → trough 0.9 = -25%
    assert abs(metrics.max_drawdown(eq) - (-0.25)) < 1e-9


def test_cagr_doubling():
    # Double over exactly 2 years (512 business days) → ~41.4% CAGR.
    eq = pd.Series(np.linspace(1.0, 2.0, 512))
    assert abs(metrics.cagr(eq) - (2**0.5 - 1)) < 0.02


def test_summary_keys():
    d = pd.Series(np.random.default_rng(1).normal(0.0003, 0.01, 1000))
    eq = (1 + d).cumprod()
    s = metrics.summary(eq, d, turnover=pd.Series(np.abs(d)))
    for k in ("sharpe", "ann_return", "ann_vol", "max_drawdown", "cagr", "ann_turnover"):
        assert k in s
