import numpy as np
import pandas as pd
import pytest

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


def test_edge_decay_worst_quartile_alarm():
    rng = np.random.default_rng(7)
    # First 1200 days: healthy positive Sharpe; last 300: decayed but still positive.
    healthy = rng.normal(0.0015, 0.007, 1200)
    decayed = rng.normal(0.0001, 0.007, 300)
    daily = pd.Series(np.concatenate([healthy, decayed]))
    # Absolute floor should not alarm (still positive).
    floor_only = monitor.edge_decay_report(daily, window=252, alarm_floor=0.0)
    assert floor_only["alarm"] is False
    # Worst-quartile flag should alarm because recent Sharpe is in bottom quartile.
    wq = monitor.edge_decay_report(
        daily, window=252, alarm_floor=0.0, alarm_on_worst_quartile=True
    )
    assert wq["worst_quartile"] is True
    assert wq["alarm"] is True


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


def test_decompose_drift_reconstructs_total():
    bt = _daily(seed=3)
    live = bt * 0.9 + 0.0002
    decomp = monitor.decompose_drift(live, bt)
    assert not decomp.get("insufficient")
    assert decomp["total_drift"] == pytest.approx(decomp["alpha"] + decomp["beta_gap"], rel=1e-6)
    assert decomp["beta"] < 1.0  # live is under-replicated


def test_implementation_shortfall_components_fallback_without_prices():
    bt = _daily(seed=4)
    live = bt + _daily(mean=0.0, vol=0.005, seed=5)
    comp = monitor.implementation_shortfall_components(live, bt)
    assert "regression" in comp
    assert "delay" not in comp
