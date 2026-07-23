"""Tests for Phase 0 (edge gate) and Phase 1 (realized-friction calibration)."""

from __future__ import annotations

import pandas as pd
import pytest

from signal_engine.config import Config
from signal_engine.data import synthetic_prices
from signal_engine.edge_gate import promotion_decision
from signal_engine.friction import (
    cost_break_even,
    load_calibration,
    net_of_friction_curve,
    realized_friction,
    slippage_bps,
    write_calibration,
)
from signal_engine.markets import symbols


# ── Phase 0: promotion policy ────────────────────────────────────────────────
class TestPromotionDecision:
    def _report(self, wf_mean_oos, folds):
        return {
            "raw": {
                "walk_forward": {
                    "mean_oos_sharpe": wf_mean_oos,
                    "folds": folds,
                }
            }
        }

    def test_promotes_on_positive_fold_delta_ci(self):
        base = self._report(0.5, [
            {"test_start": "2020-01-01", "oos_sharpe": 0.5},
            {"test_start": "2021-01-01", "oos_sharpe": 0.5},
            {"test_start": "2022-01-01", "oos_sharpe": 0.5},
        ])
        cand = self._report(0.6, [
            {"test_start": "2020-01-01", "oos_sharpe": 0.6},
            {"test_start": "2021-01-01", "oos_sharpe": 0.6},
            {"test_start": "2022-01-01", "oos_sharpe": 0.6},
        ])
        out = promotion_decision(base, cand, n_bootstrap=500, seed=1)
        assert out["verdict"] == "PROMOTE"
        assert out["backtest_promotable"] is True
        assert "fold delta CI excludes 0" in out["reason"]

    def test_holds_when_ci_spans_zero(self):
        base = self._report(0.5, [
            {"test_start": "2020-01-01", "oos_sharpe": 0.5},
            {"test_start": "2021-01-01", "oos_sharpe": 0.6},
            {"test_start": "2022-01-01", "oos_sharpe": 0.4},
        ])
        cand = self._report(0.5, [
            {"test_start": "2020-01-01", "oos_sharpe": 0.55},
            {"test_start": "2021-01-01", "oos_sharpe": 0.55},
            {"test_start": "2022-01-01", "oos_sharpe": 0.45},
        ])
        out = promotion_decision(base, cand, n_bootstrap=500, seed=1)
        assert out["verdict"] == "HOLD"
        assert out["backtest_promotable"] is False

    def test_blocks_backtest_promotion_when_pbo_high(self):
        base = self._report(0.5, [
            {"test_start": "2020-01-01", "oos_sharpe": 0.5},
            {"test_start": "2021-01-01", "oos_sharpe": 0.5},
        ])
        cand = self._report(0.6, [
            {"test_start": "2020-01-01", "oos_sharpe": 0.6},
            {"test_start": "2021-01-01", "oos_sharpe": 0.6},
        ])
        out = promotion_decision(base, cand, comparison_pbo=0.6, n_bootstrap=500, seed=1)
        assert out["verdict"] == "HOLD"
        assert "PBO" in out["reason"]

    def test_promotes_on_forward_gate_won(self):
        base = self._report(0.5, [])
        cand = self._report(0.5, [])
        # The challenger gate itself recommends promotion (>=60 days AND margin beaten).
        fwd = {
            "books": {
                "champion": {"n": 65, "sharpe": 0.4},
                "challenger_semis": {"n": 65, "sharpe": 0.75},
            },
            "recommendation": "promote:challenger_semis",
            "min_days": 60,
        }
        out = promotion_decision(
            base, cand, forward_report=fwd, candidate_book="challenger_semis"
        )
        assert out["verdict"] == "PROMOTE"
        assert out["forward_won"] is True
        assert out["forward_days"] == 65

    def test_surviving_60_days_without_winning_is_not_promotion(self):
        """A challenger that accrued 60+ days but LOST to the champion must HOLD —
        days accrued alone are not forward evidence."""
        base = self._report(0.5, [])
        cand = self._report(0.5, [])
        fwd = {
            "books": {
                "champion": {"n": 65, "sharpe": 0.6},
                "challenger_semis": {"n": 65, "sharpe": 0.1},
            },
            "recommendation": "hold",
            "min_days": 60,
        }
        out = promotion_decision(
            base, cand, forward_report=fwd, candidate_book="challenger_semis"
        )
        assert out["verdict"] == "HOLD"
        assert out["forward_won"] is False
        assert out["forward_days"] == 65


# ── Phase 1: friction ────────────────────────────────────────────────────────
class TestSlippage:
    def test_buy_above_decision_is_a_cost(self):
        assert slippage_bps(100.0, 100.5, "buy") == pytest.approx(50.0)

    def test_sell_below_decision_is_a_cost(self):
        assert slippage_bps(100.0, 99.5, "sell") == pytest.approx(50.0)

    def test_zero_price_safe(self):
        assert slippage_bps(0.0, 1.0, "buy") == 0.0


class TestRealizedFriction:
    def test_aggregates_per_symbol(self):
        fills = [
            {"symbol": "SPY", "side": "buy", "decision_price": 100, "fill_price": 100.1, "qty": 10},
            {"symbol": "SPY", "side": "sell", "decision_price": 100, "fill_price": 99.9, "qty": 10},
            {"symbol": "TLT", "side": "buy", "decision_price": 90, "fill_price": 90.18, "qty": 5},
        ]
        fr = realized_friction(fills)
        assert fr["n_fills"] == 3
        assert fr["n_symbols"] == 2
        assert "SPY" in fr["per_symbol"] and "TLT" in fr["per_symbol"]
        assert fr["round_trip_bps"] == pytest.approx(2.0 * fr["per_side_bps"])

    def test_insufficient(self):
        assert realized_friction([])["insufficient"] is True


class TestCostCurve:
    def test_net_sharpe_falls_with_cost(self):
        prices = synthetic_prices(symbols()[:6], n_days=700, seed=3)
        curve = net_of_friction_curve(prices, Config(), costs_bps=(0.0, 5.0, 20.0))
        # More cost cannot help net Sharpe.
        assert curve.iloc[0]["net_sharpe"] >= curve.iloc[-1]["net_sharpe"]
        # Gross Sharpe is ~cost-invariant (the two-pass governor couples cost in
        # only weakly via its vol estimate, so allow a small band rather than 0).
        assert curve["gross_sharpe"].max() - curve["gross_sharpe"].min() < 0.05

    def test_break_even_structure(self):
        prices = synthetic_prices(symbols()[:6], n_days=700, seed=3)
        be = cost_break_even(prices, Config(), max_cost_bps=5.0)
        assert "break_even_bps" in be and "assumed_cost_bps" in be


class TestCalibration:
    def test_write_load_and_backtest_uses_it(self, tmp_path):
        cal_path = tmp_path / "friction.json"
        fills = [
            {"symbol": "SPY", "side": "buy", "decision_price": 100, "fill_price": 100.3, "qty": 10}
        ]
        fr = realized_friction(fills)
        per_symbol = write_calibration(fr, cal_path)
        assert "SPY" in per_symbol
        loaded = load_calibration(cal_path)
        assert loaded["per_side_bps"]["SPY"] == per_symbol["SPY"]

        # A backtest with cost_scheme="calibrated" reads the file.
        from signal_engine.backtest import _cost_per_symbol

        cfg = Config(cost_scheme="calibrated", calibration_path=str(cal_path))
        costs = _cost_per_symbol(["SPY", "TLT"], cfg)
        assert costs["SPY"] == pytest.approx(per_symbol["SPY"])
        assert costs["TLT"] == pytest.approx(loaded["default_bps"])  # fallback

    def test_write_rejects_insufficient(self, tmp_path):
        with pytest.raises(ValueError):
            write_calibration({"insufficient": True}, tmp_path / "x.json")


# ── Phase 0: edge gate ───────────────────────────────────────────────────────
class TestEdgeGate:
    def test_noise_fails(self):
        from signal_engine.edge_gate import evaluate_edge

        # Driftless synthetic panel has no edge → must FAIL the hard gates.
        prices = synthetic_prices(symbols()[:8], n_days=900, seed=1)
        rep = evaluate_edge(prices, Config(), n_trials=100)
        assert rep["verdict"] == "FAIL"
        assert not all(rep["hard_gates"].values())

    def test_report_shape_and_formatting(self):
        from signal_engine.edge_gate import evaluate_edge, format_verdict

        prices = synthetic_prices(symbols()[:8], n_days=900, seed=1)
        rep = evaluate_edge(prices, Config(), n_trials=100)
        assert set(rep["hard_gates"]) == {
            "H1_clears_noise",
            "H2_edge_real",
            "H3_passes_deflated",
            "H4_beats_buy_hold",
        }
        assert set(rep["robustness_gates"]) == {
            "R1_cpcv_robust",
            "R2_walk_forward_ok",
            "R3_cost_headroom",
        }
        assert rep["verdict"] in format_verdict(rep)

    def test_forward_track_included_when_live_data(self):
        from signal_engine.edge_gate import evaluate_edge

        prices = synthetic_prices(symbols()[:8], n_days=900, seed=1)
        live = pd.Series(
            0.0005,
            index=pd.date_range("2025-01-01", periods=40, freq="B"),
        )
        rep = evaluate_edge(prices, Config(), n_trials=100, live_returns=live)
        assert rep["forward_track"] is not None
        assert rep["forward_track"]["n_days"] == 40

    def test_h4_beats_buy_hold_populates_benchmark_metrics(self):
        from signal_engine.edge_gate import evaluate_edge

        prices = synthetic_prices(symbols()[:8], n_days=900, seed=1)
        rep = evaluate_edge(prices, Config(), n_trials=100)
        assert "H4_beats_buy_hold" in rep["hard_gates"]
        assert rep["metrics"]["benchmark_symbol"] == "SPY"
        assert rep["metrics"]["strategy_cagr"] is not None
        assert rep["metrics"]["benchmark_cagr"] is not None

    def test_h4_false_when_benchmark_symbol_absent(self):
        from signal_engine.edge_gate import evaluate_edge

        # Universe without SPY at all -> benchmark comparison is skipped, H4 fails closed.
        prices = synthetic_prices(["IWM", "EFA", "EEM", "EWJ", "TLT", "IEF", "TIP", "GLD"], n_days=900, seed=1)
        rep = evaluate_edge(prices, Config(), n_trials=100, benchmark_symbol="SPY")
        assert rep["hard_gates"]["H4_beats_buy_hold"] is False
        assert rep["metrics"]["strategy_cagr"] is None
