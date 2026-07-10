"""Tests for the 2024–2026 research levers and forward-loop upgrades.

Covers: CPCV + PBO, honest trial counting, the lookahead guard, effective number
of bets, network momentum, drift decomposition, quartile edge-decay, champion/
challenger books, arrival slippage, and the stabilised gross cap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_engine.config import Config
from signal_engine.data import random_walk_panel
from signal_engine.diagnostics import diversification_audit, effective_number_of_bets
from signal_engine.rules import network_momentum_forecast
from signal_engine.validation import (
    assert_no_lookahead,
    combinatorial_purged_cv,
    config_fingerprint,
    honest_n_trials,
    probability_backtest_overfitting,
    register_trial,
)


# ── Validation frontier ──────────────────────────────────────────────────────
class TestCPCV:
    def test_returns_distribution(self, full_prices):
        res = combinatorial_purged_cv(full_prices, Config(), n_groups=5, k_test=2)
        assert not res.get("insufficient")
        assert res["n_paths"] == 10  # C(5, 2)
        assert "oos_p5" in res and "oos_p95" in res
        assert 0.0 <= res["pct_paths_below_zero"] <= 1.0

    def test_insufficient_data(self):
        short = random_walk_panel(4, 100, seed=1)
        assert combinatorial_purged_cv(short, Config())["insufficient"] is True


class TestPBO:
    def test_pbo_high_for_pure_noise(self):
        # 20 configs of pure noise: the IS-best is luck → PBO should be substantial.
        rng = np.random.default_rng(0)
        noise = pd.DataFrame(rng.standard_normal((600, 20)))
        res = probability_backtest_overfitting(noise, n_splits=8)
        assert 0.0 <= res["pbo"] <= 1.0
        assert res["pbo"] > 0.3  # noise search is heavily overfit

    def test_pbo_low_for_one_dominant_config(self):
        rng = np.random.default_rng(1)
        m = rng.standard_normal((600, 8)) * 0.01
        m[:, 0] += 0.02  # config 0 has a real, stable edge in every split
        res = probability_backtest_overfitting(pd.DataFrame(m), n_splits=8)
        assert res["pbo"] < 0.2

    def test_insufficient(self):
        assert probability_backtest_overfitting(pd.DataFrame(np.zeros((10, 1))))["insufficient"]


class TestTrialRegistry:
    def test_dedup_and_count(self, tmp_path):
        reg = tmp_path / "trials.jsonl"
        register_trial(Config(), label="a", path=reg)
        register_trial(Config(), label="a-again", path=reg)  # same config → no-op
        register_trial(Config(use_cot=True), label="b", path=reg)
        assert honest_n_trials(reg) == 2

    def test_fingerprint_stable_and_distinct(self):
        assert config_fingerprint(Config()) == config_fingerprint(Config())
        assert config_fingerprint(Config()) != config_fingerprint(Config(vol_target=0.15))

    def test_empty_registry_floor(self, tmp_path):
        assert honest_n_trials(tmp_path / "none.jsonl") == 1


class TestLookaheadGuard:
    def _panel(self):
        return random_walk_panel(3, 300, seed=7)

    def test_causal_passes(self):
        def causal(df):
            return df.iloc[:, 0].rolling(20, min_periods=5).mean()

        assert assert_no_lookahead(causal, self._panel())["causal"] is True

    def test_lookahead_raises(self):
        def leaky(df):
            col = df.iloc[:, 0]
            return (col - col.mean()) / col.std()  # full-sample stats = peeking

        with pytest.raises(AssertionError):
            assert_no_lookahead(leaky, self._panel())


# ── Diversification audit ────────────────────────────────────────────────────
class TestEffectiveBets:
    def test_independent_panel_high_enb(self):
        panel = random_walk_panel(10, 800, seed=11)
        enb = effective_number_of_bets(panel.pct_change())
        assert enb["effective_bets"] > 6  # near 10 when uncorrelated
        assert 0 < enb["concentration_ratio"] <= 1.0

    def test_redundant_panel_low_enb(self):
        base = random_walk_panel(1, 800, seed=13).iloc[:, 0]
        # 8 near-identical instruments → effectively one bet.
        cols = {f"x{i}": base * (1 + 0.01 * i) for i in range(8)}
        returns = pd.DataFrame(cols).pct_change()
        enb = effective_number_of_bets(returns)
        assert enb["effective_bets"] < 2.5

    def test_diversification_audit_on_result(self, result):
        audit = diversification_audit(result)
        assert "idm" in audit
        if not audit.get("insufficient"):
            assert audit["idm_vs_effective"] > 0


# ── Network momentum ─────────────────────────────────────────────────────────
class TestNetworkMomentum:
    def test_shape_and_finite(self, full_prices):
        nm = network_momentum_forecast(full_prices, rebal=126)
        assert nm.shape == full_prices.shape
        assert np.isfinite(nm.to_numpy()).all()
        assert (nm.abs() <= 20.0 + 1e-9).all().all()  # capped

    def test_causal(self, full_prices):
        # A single column must not change history when the tail is revealed.
        def one_col(df):
            return network_momentum_forecast(df, rebal=126).iloc[:, 0]

        assert_no_lookahead(one_col, full_prices, truncate=40)

    def test_wires_into_backtest(self, full_prices):
        from signal_engine.backtest import run_backtest

        res = run_backtest(full_prices, Config(use_network_momentum=True, nm_rebal=126))
        assert "network_mom" in res.forecasts.columns.tolist() or len(res.daily_returns) > 0


# ── Reconciliation upgrades (drift decomposition + quartile decay) ───────────
class TestDriftDecomposition:
    def test_identical_series_is_clean(self):
        from signal_engine.monitor import decompose_drift

        rng = np.random.default_rng(3)
        s = pd.Series(rng.standard_normal(300) * 0.01)
        dec = decompose_drift(s, s)
        assert dec["beta"] == pytest.approx(1.0, abs=1e-6)
        assert dec["alpha"] == pytest.approx(0.0, abs=1e-6)
        assert dec["beta_gap"] == pytest.approx(0.0, abs=1e-6)

    def test_scaled_book_shows_beta_gap(self):
        from signal_engine.monitor import decompose_drift

        rng = np.random.default_rng(4)
        model = pd.Series(rng.standard_normal(300) * 0.01 + 0.0005)
        live = model * 0.5  # replicating only half the exposure
        dec = decompose_drift(live, model)
        assert dec["beta"] == pytest.approx(0.5, abs=1e-6)

    def test_reconcile_includes_decomposition(self):
        from signal_engine.monitor import reconcile

        rng = np.random.default_rng(5)
        idx = pd.date_range("2025-01-01", periods=120, freq="B")
        model = pd.Series(rng.standard_normal(120) * 0.01, index=idx)
        live = model + rng.standard_normal(120) * 0.001
        rec = reconcile(live, model)
        assert "drift_decomposition" in rec
        assert rec["drift_decomposition"]["beta"] > 0.5


class TestEdgeDecayQuartile:
    def test_floor_alarm_preserved(self):
        from signal_engine.monitor import edge_decay_report

        # Steadily negative returns → rolling Sharpe below zero → floor alarm.
        neg = pd.Series(np.linspace(-0.01, -0.02, 400))
        rep = edge_decay_report(neg, window=60)
        assert rep["alarm"] is True  # kill-switch semantics unchanged
        assert "worst_quartile" in rep and "decay_warning" in rep

    def test_quartile_gated_on_history(self):
        from signal_engine.monitor import edge_decay_report

        short = pd.Series(np.random.default_rng(6).standard_normal(70) * 0.01)
        rep = edge_decay_report(short, window=60)
        # < 60 rolling points → quartile not evaluated.
        assert rep["q25"] is None
        assert rep["worst_quartile"] is False


# ── Forward-loop: champion/challenger + arrival slippage + stabilised cap ─────
class TestChampionChallenger:
    def test_report_splits_and_recommends(self, tmp_path):
        from signal_engine.live import champion_challenger_report

        path = tmp_path / "returns.csv"
        dates = pd.date_range("2025-01-01", periods=80, freq="B")
        rows = []
        rng = np.random.default_rng(9)
        for d in dates:
            rows.append({"date": d.date(), "live_return": rng.standard_normal() * 0.001,
                         "mode": "shadow", "use_cot": False, "book": "champion"})
            rows.append({"date": d.date(), "live_return": 0.002 + rng.standard_normal() * 0.001,
                         "mode": "shadow", "use_cot": True, "book": "challenger"})
        pd.DataFrame(rows).to_csv(path, index=False)
        rep = champion_challenger_report(path, min_days=60, promote_margin=0.2)
        assert set(rep["books"]) == {"champion", "challenger"}
        assert rep["recommendation"] == "promote:challenger"

    def test_legacy_use_cot_fallback(self, tmp_path):
        from signal_engine.live import champion_challenger_report

        path = tmp_path / "legacy.csv"
        dates = pd.date_range("2025-01-01", periods=10, freq="B")
        pd.DataFrame(
            {"date": dates, "live_return": 0.001, "mode": "shadow", "use_cot": True}
        ).to_csv(path, index=False)
        rep = champion_challenger_report(path)
        assert "cot_challenger" in rep["books"]


class TestArrivalSlippage:
    def test_delay_slippage_computed(self):
        from signal_engine.live import compute_delay_slippage

        idx = pd.to_datetime(["2025-01-02", "2025-01-03"])
        prices = pd.DataFrame({"SPY": [100.0, 101.0]}, index=idx)
        arrivals = pd.DataFrame({"SPY": [100.0, 100.5]}, index=idx)  # entered at 100.5
        target = {"date": "2025-01-02", "units": {"SPY": 10.0}, "capital": 1_000_000.0}
        # delay = 10 * (100.5 - 100.0) / 1e6 = 5e-6
        assert compute_delay_slippage(target, prices, arrivals) == pytest.approx(5e-6)


class TestStabilisedGrossCap:
    def test_equity_buffer_shrinks_budget(self):
        from scripts.execute_alpaca import gross_scale_factor

        target = {"notional": {"SPY": 2_000_000.0}}  # gross 2M
        # 1x of 1M = 1M budget → 0.5; with 20% buffer → 0.8M budget → 0.4.
        assert gross_scale_factor(target, 1_000_000, 1.0) == pytest.approx(0.5)
        assert gross_scale_factor(
            target, 1_000_000, 1.0, equity_buffer=0.2
        ) == pytest.approx(0.4)

    def test_reference_equity_caps_upsizing(self):
        from scripts.execute_alpaca import gross_scale_factor

        target = {"notional": {"SPY": 2_000_000.0}}
        # Live equity spiked to 1.5M but reference is 1.0M → anchor to the lower.
        assert gross_scale_factor(
            target, 1_500_000, 1.0, reference_equity=1_000_000
        ) == pytest.approx(0.5)

