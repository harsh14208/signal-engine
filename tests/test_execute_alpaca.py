"""Tests for the Alpaca paper executor: gross-exposure cap and zero-cross safety."""

from __future__ import annotations

from unittest.mock import patch

import pytest


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
