from signal_engine.markets import asset_classes, symbols
from signal_engine.weights import cluster_weights


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
