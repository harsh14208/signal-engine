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


def test_benchmark_comparison_detects_beat_and_loss():
    idx = pd.date_range("2015-01-01", periods=1000, freq="B")
    # Strategy clearly beats a low-drift benchmark.
    strat_daily = pd.Series(0.0008, index=idx)
    strat_equity = (1 + strat_daily).cumprod()
    weak_bench = pd.Series(100.0, index=idx) * (1 + pd.Series(0.0001, index=idx)).cumprod()
    res = metrics.benchmark_comparison(strat_equity, strat_daily, weak_bench)
    assert res["beats_cagr"] and res["beats_sharpe"] and res["beats_both"]
    assert res["beats_priority"]

    # Same strategy clearly loses to a much stronger benchmark, on a real gap
    # (not just close-but-better-drawdown) -> fails the priority check too.
    strong_bench = pd.Series(100.0, index=idx) * (1 + pd.Series(0.002, index=idx)).cumprod()
    res2 = metrics.benchmark_comparison(strat_equity, strat_daily, strong_bench)
    assert not res2["beats_cagr"] and not res2["beats_both"]
    assert not res2["beats_priority"]


def test_benchmark_comparison_priority_rescues_close_cagr_with_better_drawdown():
    """The whole point of the CAGR>MaxDD>Sharpe priority: a strategy that nearly
    matches buy-and-hold CAGR while drawing down much less should PASS, even
    though a naive "must beat CAGR outright" check would fail it."""
    rng = np.random.default_rng(3)
    idx = pd.date_range("2007-01-01", periods=2000, freq="B")

    # Strategy: modest drift, low vol -> small drawdowns, CAGR close to bench.
    strat_daily = pd.Series(rng.normal(0.00035, 0.004, len(idx)), index=idx)
    strat_equity = (1 + strat_daily).cumprod()

    # Benchmark: similar drift but with an occasional large negative shock ->
    # a much deeper drawdown, like a real equity index crash.
    bench_daily = rng.normal(0.00035, 0.004, len(idx))
    bench_daily[500] = -0.20  # one crash day
    bench_prices = pd.Series(100.0, index=idx) * (1 + pd.Series(bench_daily, index=idx)).cumprod()

    res = metrics.benchmark_comparison(strat_equity, strat_daily, bench_prices)
    assert res["cagr_close"]
    assert res["maxdd_meaningfully_better"]
    assert res["beats_priority"]
    assert abs(res["strategy_max_drawdown"]) < abs(res["benchmark_max_drawdown"])


def test_benchmark_comparison_insufficient_when_no_overlap():
    idx = pd.date_range("2015-01-01", periods=10, freq="B")
    strat_daily = pd.Series(0.001, index=idx)
    strat_equity = (1 + strat_daily).cumprod()
    other_idx = pd.date_range("2020-01-01", periods=10, freq="B")
    bench = pd.Series(100.0, index=other_idx)
    res = metrics.benchmark_comparison(strat_equity, strat_daily, bench)
    assert res["insufficient"]
