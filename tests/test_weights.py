import numpy as np
import pandas as pd

from signal_engine.config import Config
from signal_engine.markets import asset_classes, instrument_for, symbols
from signal_engine.weights import (
    build_instrument_weights,
    cluster_weights,
    corr_cluster_weights,
    equal_weights,
    sharpe_adjusted_weights,
)


def test_weights_sum_to_one():
    w = cluster_weights(symbols())
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_each_cluster_gets_equal_share():
    classes = asset_classes()
    n = len(classes)
    w = cluster_weights(symbols())
    for members in classes.values():
        assert abs(sum(w[m] for m in members) - 1.0 / n) < 1e-9


def test_within_cluster_equal():
    w = cluster_weights(symbols())
    for members in asset_classes().values():
        vals = [w[m] for m in members]
        assert max(vals) - min(vals) < 1e-9


def test_cluster_deconcentrates_equity():
    # The 5 correlated equity ETFs should get LESS than naive 1/N would give them.
    syms = symbols()
    w = cluster_weights(syms)
    eq = asset_classes()["equity"]
    eq_total = sum(w[s] for s in eq)
    equal_total = len(eq) / len(syms)  # 5/19 ≈ 0.26
    assert eq_total < equal_total  # cluster gives the sleeve 1/6 ≈ 0.17


def test_corr_cluster_weights_sum_to_one():
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(
        {
            "a": rng.normal(0, 0.01, 200),
            "b": rng.normal(0, 0.01, 200),
            "c": rng.normal(0, 0.01, 200),
        }
    )
    w = corr_cluster_weights(["a", "b", "c"], rets, threshold=0.5)
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_corr_cluster_does_not_inflate_singleton():
    # a and b are highly correlated; c is independent.  Each cluster should get
    # 50% of risk, so c gets 0.5 — not the 1.0 it would receive if it were its
    # own asset-class cluster.
    rng = np.random.default_rng(1)
    common = rng.normal(0, 0.01, size=200)
    rets = pd.DataFrame(
        {
            "a": common + rng.normal(0, 0.001, 200),
            "b": common + rng.normal(0, 0.001, 200),
            "c": rng.normal(0, 0.01, 200),
        }
    )
    w = corr_cluster_weights(["a", "b", "c"], rets, threshold=0.5)
    assert abs(w["a"] - w["b"]) < 1e-9
    assert abs(w["a"] + w["b"] - 0.5) < 1e-9
    assert abs(w["c"] - 0.5) < 1e-9


def test_sharpe_adjusted_down_weights_negative_sharpe():
    # a and b have positive Sharpe; c is strongly negative.
    idx = pd.date_range("2015-01-01", periods=200)
    rets = pd.DataFrame(
        {
            "a": np.full(200, 0.0004),
            "b": np.full(200, 0.0003),
            "c": np.full(200, -0.0005),
        },
        index=idx,
    )
    # Add tiny noise so std > 0.
    rng = np.random.default_rng(2)
    rets += rng.normal(0, 0.005, size=rets.shape)
    w = sharpe_adjusted_weights(["a", "b", "c"], rets)
    assert w["c"] < w["a"]
    assert w["c"] < w["b"]
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_build_instrument_weights_dispatch():
    rng = np.random.default_rng(3)
    rets = pd.DataFrame(rng.normal(0, 0.01, size=(200, 4)), columns=list("abcd"))
    eq = build_instrument_weights(list("abcd"), rets, Config(weight_scheme="equal"))
    cc = build_instrument_weights(list("abcd"), rets, Config(weight_scheme="corr_cluster"))
    sh = build_instrument_weights(list("abcd"), rets, Config(weight_scheme="sharpe"))
    for w in (eq, cc, sh):
        assert abs(sum(w.values()) - 1.0) < 1e-9
    assert max(eq.values()) - min(eq.values()) < 1e-9


def test_equal_weights():
    w = equal_weights(["x", "y", "z"])
    assert w == {"x": 1 / 3, "y": 1 / 3, "z": 1 / 3}


def test_instrument_for_expanded_universe():
    # FXI is only in the expanded universe.
    assert instrument_for("FXI", expanded=False) is None
    assert instrument_for("FXI", expanded=True) is not None
    assert instrument_for("FXI", expanded=True).asset_class == "equity"


def test_cluster_weights_use_expanded_metadata():
    # FXI and SPY are both equities in the expanded universe → one cluster.
    w_expanded = cluster_weights(["FXI", "SPY"], expanded=True)
    assert abs(w_expanded["FXI"] - 0.5) < 1e-9
    assert abs(w_expanded["SPY"] - 0.5) < 1e-9

    # With core metadata FXI is unknown → treated as a separate cluster.
    w_core = cluster_weights(["FXI", "SPY"], expanded=False)
    assert abs(w_core["FXI"] - 0.5) < 1e-9
    assert abs(w_core["SPY"] - 0.5) < 1e-9


def test_build_instrument_weights_expanded_flag():
    rng = np.random.default_rng(4)
    rets = pd.DataFrame(rng.normal(0, 0.01, size=(200, 2)), columns=["FXI", "SPY"])
    cfg = Config(weight_scheme="cluster", use_expanded_universe=True)
    w = build_instrument_weights(["FXI", "SPY"], rets, cfg)
    # Both equities under expanded metadata → equal split.
    assert abs(w["FXI"] - w["SPY"]) < 1e-9
