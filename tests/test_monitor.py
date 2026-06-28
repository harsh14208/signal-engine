import numpy as np
import pandas as pd

from signal_engine import monitor


def _daily(mean=0.0004, vol=0.01, n=1500, seed=0):
    return pd.Series(np.random.default_rng(seed).normal(mean, vol, n))


def test_rolling_sharpe_length_and_warmup():
    d = _daily()
    rs = monitor.rolling_sharpe(d, window=252)
    assert rs.iloc[:251].isna().all()
    assert rs.iloc[252:].notna().all()


def test_edge_decay_report_keys_and_alarm():
    # A strongly positive series → no alarm (last window reliably > 0).
    rep = monitor.edge_decay_report(_daily(mean=0.0015, vol=0.007), window=252)
    for k in ("current", "median", "min", "max", "pct_windows_below_zero", "alarm"):
        assert k in rep
    assert rep["alarm"] is False

    # A clearly negative-drift series → alarm fires.
    bad = monitor.edge_decay_report(_daily(mean=-0.0012, vol=0.007), window=252)
    assert bad["alarm"] is True


def test_reconcile_perfect_match():
    bt = _daily(seed=1)
    rep = monitor.reconcile(bt.copy(), bt.copy())
    assert rep["corr"] > 0.99
    assert rep["tracking_error"] < 1e-6
    assert rep["aligned"] is True


def test_reconcile_detects_divergence():
    bt = _daily(seed=2)
    live = bt + _daily(mean=0.0, vol=0.01, seed=99)  # uncorrelated noise on top
    rep = monitor.reconcile(live, bt)
    assert rep["corr"] < 0.9
    assert rep["aligned"] is False


def test_reconcile_insufficient():
    s = pd.Series([0.01, -0.01, 0.0])
    assert monitor.reconcile(s, s).get("insufficient")
