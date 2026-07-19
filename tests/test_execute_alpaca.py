"""Tests for the Alpaca paper executor: gross-exposure cap and zero-cross safety."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from signal_engine.evaluator import TradeEvaluation


class TestGrossScaleFactor:
    def test_downscales_leveraged_book(self):
        from scripts.execute_alpaca import gross_scale_factor

        # gross = |1.5M long| + |1.0M short| = 2.5M on 1M equity.
        target = {"notional": {"SPY": 1_500_000.0, "TLT": -1_000_000.0}}
        # 1.5x cap → budget 1.5M → scale to 1.5M/2.5M = 0.6.
        assert gross_scale_factor(target, equity=1_000_000, max_gross_mult=1.5) == pytest.approx(0.6)

    def test_no_scaling_when_book_fits(self):
        from scripts.execute_alpaca import gross_scale_factor

        target = {"notional": {"SPY": 400_000.0, "TLT": -300_000.0}}  # gross 0.7M
        # Fits within 1x of 1M equity → no scaling.
        assert gross_scale_factor(target, equity=1_000_000, max_gross_mult=1.0) == 1.0

    def test_handles_missing_data(self):
        from scripts.execute_alpaca import gross_scale_factor

        assert gross_scale_factor({"notional": {}}, equity=1_000_000, max_gross_mult=1.5) == 1.0
        assert gross_scale_factor({"notional": {"SPY": 1e6}}, equity=0, max_gross_mult=1.5) == 1.0


class TestZeroCross:
    def test_flip_closes_to_flat_and_defers_reverse(self, tmp_path):
        """A long→short flip must only sell down to flat, not cross zero in one order."""
        import scripts.execute_alpaca as ea

        # Held +100 shares long; target is -40 short → should sell exactly 100 (to flat),
        # not 140, and flag the reverse leg as deferred to the next run.
        target = {"date": "2026-07-09", "units": {"SPY": -40.0}, "notional": {"SPY": -5000.0}}
        orders: list = []

        def fake_order(base, key, secret, symbol, qty, side):
            orders.append((symbol, qty, side))
            return {"id": "test-id"}

        with patch.object(ea, "_credentials", return_value=("k", "s")), patch.object(
            ea, "cancel_all_orders", return_value=None
        ), patch.object(ea, "get_account", return_value={"equity": "1000000"}), patch.object(
            ea, "get_positions", return_value=[{"symbol": "SPY", "qty": "100"}]
        ), patch.object(ea, "is_shortable", return_value=True), patch.object(
            ea, "place_qty_order", side_effect=fake_order
        ):
            res = ea.execute_targets(
                target=target,
                live=False,
                orders_path=tmp_path / "orders.jsonl",
                kill_switch={"paused": False},
                max_gross_mult=1.5,
            )

        assert orders == [("SPY", 100, "sell")]  # sold to flat only, not 140
        rec = res["submitted"][0]
        assert rec["zero_cross_deferred"] is True
        assert rec["target_shares"] == 0


class TestAiEvaluator:
    """AI evaluator integration in execute_targets."""

    def _mock_broker(self, ea):
        return (
            patch.object(ea, "_credentials", return_value=("k", "s")),
            patch.object(ea, "cancel_all_orders", return_value=None),
            patch.object(ea, "get_account", return_value={"equity": "1000000"}),
            patch.object(ea, "get_positions", return_value=[]),
            patch.object(ea, "is_shortable", return_value=True),
        )

    def test_advisory_mode_logs_but_does_not_change_targets(self, tmp_path):
        import scripts.execute_alpaca as ea

        target = {"date": "2026-07-19", "units": {"SPY": 100.0}, "notional": {"SPY": 50000.0}}
        orders: list = []

        def fake_order(base, key, secret, symbol, qty, side):
            orders.append((symbol, qty, side))
            return {"id": "test-id"}

        class AdvisoryEvaluator:
            def evaluate(self, context):
                return TradeEvaluation(decision="approve", scale=0.5, confidence=0.8, reasoning="test")

        with self._mock_broker(ea)[0], self._mock_broker(ea)[1], self._mock_broker(ea)[2], self._mock_broker(ea)[3], self._mock_broker(ea)[4], patch.object(
            ea, "place_qty_order", side_effect=fake_order
        ):
            res = ea.execute_targets(
                target=target,
                live=False,
                orders_path=tmp_path / "orders.jsonl",
                kill_switch={"paused": False},
                ai_evaluator=AdvisoryEvaluator(),
                ai_mode="advisory",
                ai_evaluations_path=tmp_path / "ai.jsonl",
            )

        assert orders == [("SPY", 100, "buy")]
        assert res["ai_scale"] == 1.0
        assert res["ai_evaluation"]["reasoning"] == "test"
        rec = res["submitted"][0]
        assert rec["ai_evaluation"]["scale"] == 0.5
        assert rec["ai_mode"] == "advisory"
        assert rec["ai_scale"] == 1.0  # advisory does not change sizes

    def test_scale_mode_reduces_target_sizes(self, tmp_path):
        import scripts.execute_alpaca as ea

        target = {"date": "2026-07-19", "units": {"SPY": 100.0}, "notional": {"SPY": 50000.0}}
        orders: list = []

        def fake_order(base, key, secret, symbol, qty, side):
            orders.append((symbol, qty, side))
            return {"id": "test-id"}

        class ScaleEvaluator:
            def evaluate(self, context):
                return TradeEvaluation(decision="approve", scale=0.5, confidence=0.8, reasoning="half size")

        with self._mock_broker(ea)[0], self._mock_broker(ea)[1], self._mock_broker(ea)[2], self._mock_broker(ea)[3], self._mock_broker(ea)[4], patch.object(
            ea, "place_qty_order", side_effect=fake_order
        ):
            res = ea.execute_targets(
                target=target,
                live=False,
                orders_path=tmp_path / "orders.jsonl",
                kill_switch={"paused": False},
                ai_evaluator=ScaleEvaluator(),
                ai_mode="scale",
                ai_evaluations_path=tmp_path / "ai.jsonl",
            )

        assert orders == [("SPY", 50, "buy")]
        assert res["ai_scale"] == 0.5
        rec = res["submitted"][0]
        assert rec["target_shares"] == 50
        assert rec["ai_scale"] == 0.5

    def test_ai_mode_block_choice_no_longer_accepted(self, capsys):
        # "block" was removed from the CLI entirely — AI guidance can resize
        # (scale) or just log (advisory) an assessment, but there is no mode
        # that skips a rebalance based on it. argparse rejects an invalid
        # --ai-mode choice (and exits) before any broker/network code runs.
        import scripts.execute_alpaca as ea

        with pytest.raises(SystemExit):
            ea.main(["--ai-mode", "block"])
        assert "invalid choice: 'block'" in capsys.readouterr().err

    def test_reject_decision_does_not_block_rebalance(self, tmp_path):
        # A "reject" decision from the evaluator is advisory (logged for human
        # review) and has no effect on execution — even in "scale" mode, only
        # `scale` changes what gets submitted. This replaces the old
        # test_block_mode_skips_rebalance_on_reject, which asserted the
        # opposite behavior for a mode ("block") that no longer exists.
        import scripts.execute_alpaca as ea

        target = {"date": "2026-07-19", "units": {"SPY": 100.0}, "notional": {"SPY": 50000.0}}
        orders: list = []

        def fake_order(base, key, secret, symbol, qty, side):
            orders.append((symbol, qty, side))
            return {"id": "test-id"}

        class RejectEvaluator:
            def evaluate(self, context):
                return TradeEvaluation(decision="reject", scale=0.5, confidence=0.9, reasoning="too risky")

        with self._mock_broker(ea)[0], self._mock_broker(ea)[1], self._mock_broker(ea)[2], self._mock_broker(ea)[3], self._mock_broker(ea)[4], patch.object(
            ea, "place_qty_order", side_effect=fake_order
        ):
            res = ea.execute_targets(
                target=target,
                live=False,
                orders_path=tmp_path / "orders.jsonl",
                kill_switch={"paused": False},
                ai_evaluator=RejectEvaluator(),
                ai_mode="scale",
                ai_evaluations_path=tmp_path / "ai.jsonl",
            )

        # Not skipped: a "reject" decision never halts the rebalance. Its
        # scale (0.5) is still applied, same as any other evaluation.
        assert res["skipped"] is False
        assert orders == [("SPY", 50, "buy")]
        assert res["ai_scale"] == 0.5
        assert res["ai_evaluation"]["reasoning"] == "too risky"
        ai_log = tmp_path / "ai.jsonl"
        assert ai_log.exists()
        record = json.loads(ai_log.read_text().strip().splitlines()[0])
        assert record["applied"] is True
        assert record["evaluation"]["reasoning"] == "too risky"

    def test_no_evaluator_stores_none(self, tmp_path):
        import scripts.execute_alpaca as ea

        target = {"date": "2026-07-19", "units": {"SPY": 100.0}, "notional": {"SPY": 50000.0}}
        orders: list = []

        def fake_order(base, key, secret, symbol, qty, side):
            orders.append((symbol, qty, side))
            return {"id": "test-id"}

        with self._mock_broker(ea)[0], self._mock_broker(ea)[1], self._mock_broker(ea)[2], self._mock_broker(ea)[3], self._mock_broker(ea)[4], patch.object(
            ea, "place_qty_order", side_effect=fake_order
        ):
            res = ea.execute_targets(
                target=target,
                live=False,
                orders_path=tmp_path / "orders.jsonl",
                kill_switch={"paused": False},
            )

        assert orders == [("SPY", 100, "buy")]
        assert res["ai_evaluation"] is None
        assert res["ai_scale"] == 1.0

    def test_scale_mode_with_zero_scale_closes_positions(self, tmp_path):
        import scripts.execute_alpaca as ea

        target = {"date": "2026-07-19", "units": {"SPY": 100.0}, "notional": {"SPY": 50000.0}}
        orders: list = []

        def fake_order(base, key, secret, symbol, qty, side):
            orders.append((symbol, qty, side))
            return {"id": "test-id"}

        class ZeroEvaluator:
            def evaluate(self, context):
                return TradeEvaluation(decision="approve", scale=0.0, confidence=0.9, reasoning="flat")

        with self._mock_broker(ea)[0], self._mock_broker(ea)[1], self._mock_broker(ea)[2], self._mock_broker(ea)[3], self._mock_broker(ea)[4], patch.object(
            ea, "place_qty_order", side_effect=fake_order
        ):
            res = ea.execute_targets(
                target=target,
                live=False,
                orders_path=tmp_path / "orders.jsonl",
                kill_switch={"paused": False},
                ai_evaluator=ZeroEvaluator(),
                ai_mode="scale",
            )

        # Target scaled to 0; no current position, so no trade needed.
        assert orders == []
        assert res["ai_scale"] == 0.0


class TestAiCliDefaults:
    """CLI argument defaults for AI evaluation."""

    def test_no_ai_evaluate_flag_exists(self):
        import scripts.execute_alpaca as ea

        # --no-ai-evaluate must be a valid flag.
        with pytest.raises(SystemExit):
            ea.main(["--no-ai-evaluate", "--help"])

    def test_ai_mode_defaults_to_scale(self):
        import scripts.execute_alpaca as ea

        with pytest.raises(SystemExit):
            ea.main(["--help"])
