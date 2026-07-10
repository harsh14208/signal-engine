"""Phase 2 tests — market calendar, data-quality audit, feature store, lineage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_engine import market_calendar as mc
from signal_engine.data_quality import (
    audit_panel,
    audit_symbol,
    health_scorecard,
    record_fetch,
)
from signal_engine.feature_store import (
    list_snapshots,
    load_snapshot,
    save_snapshot,
    snapshot_hash,
)
from signal_engine.lineage import lineage_hash, universe_hash


class TestCalendar:
    def test_session_count_reasonable(self):
        # NYSE trades ~250–252 days/year.
        assert 249 <= len(mc.trading_days("2023-01-01", "2023-12-31")) <= 253

    def test_holidays_excluded(self):
        assert not mc.is_trading_day("2023-12-25")  # Christmas
        assert not mc.is_trading_day("2023-07-04")  # Independence Day
        assert not mc.is_trading_day("2023-01-14")  # Saturday

    def test_ad_hoc_closure(self):
        assert not mc.is_trading_day("2012-10-30")  # Hurricane Sandy
        assert not mc.is_trading_day("2025-01-09")  # Carter mourning

    def test_missing_sessions_detects_gap(self):
        full = mc.trading_days("2023-03-01", "2023-03-31")
        gapped = full.delete([5, 6])  # drop two real sessions
        missing = mc.missing_sessions(gapped)
        assert full[5] in missing and full[6] in missing

    def test_next_prev(self):
        # 2023-01-16 is MLK day (holiday); Friday 13th → next is Tuesday 17th.
        assert mc.next_trading_day("2023-01-13") == pd.Timestamp("2023-01-17")
        assert mc.previous_trading_day("2023-01-17") == pd.Timestamp("2023-01-13")


class TestDataQuality:
    def _clean_series(self, n=300):
        idx = mc.trading_days("2022-01-01", "2024-12-31")[:n]
        return pd.Series(100 + np.arange(n) * 0.1, index=idx)

    def test_clean_series_scores_high(self):
        rep = audit_symbol(self._clean_series())
        assert rep["health_score"] >= 90
        assert rep["jumps"] == 0

    def test_jump_penalised(self):
        s = self._clean_series()
        s.iloc[150] *= 2.0  # +100% print → unadjusted-split style jump
        rep = audit_symbol(s)
        assert rep["jumps"] >= 1
        assert rep["health_score"] < 90

    def test_flatline_penalised(self):
        s = self._clean_series()
        s.iloc[100:110] = s.iloc[100]  # dead feed
        rep = audit_symbol(s)
        assert rep["max_flatline"] >= 5
        assert rep["health_score"] < 90

    def test_panel_audit(self):
        idx = mc.trading_days("2022-01-01", "2024-12-31")[:400]
        panel = pd.DataFrame(
            {"A": 100 + np.arange(400) * 0.1, "B": 50 + np.arange(400) * 0.05}, index=idx
        )
        rep = audit_panel(panel, end=idx[-1].strftime("%Y-%m-%d"))
        assert rep["n_symbols"] == 2
        assert rep["mean_health"] >= 90
        assert rep["healthy"] is True

    def test_fetch_telemetry(self, tmp_path):
        log = tmp_path / "health.jsonl"
        record_fetch("yfinance", ok=True, n_rows=100, latency_ms=120, path=log)
        record_fetch("yfinance", ok=False, note="timeout", path=log)
        card = health_scorecard(log)
        assert card["yfinance"]["n_calls"] == 2
        assert card["yfinance"]["error_rate"] == pytest.approx(0.5)


class TestFeatureStore:
    def test_save_load_roundtrip(self, tmp_path):
        feats = {"date": "2026-07-09", "forecast": {"SPY": 3.2}, "idm": 2.1, "bad": float("nan")}
        env = save_snapshot("2026-07-09", feats, snapshot_dir=tmp_path)
        assert env["content_hash"]
        assert env["lineage_hash"] == lineage_hash()
        loaded = load_snapshot("2026-07-09", snapshot_dir=tmp_path)
        assert loaded["features"]["bad"] is None  # NaN → None
        assert "2026-07-09_champion" in list_snapshots(tmp_path)

    def test_immutability_conflict(self, tmp_path):
        save_snapshot("2026-07-09", {"x": 1}, snapshot_dir=tmp_path)
        save_snapshot("2026-07-09", {"x": 1}, snapshot_dir=tmp_path)  # identical → no-op
        with pytest.raises(ValueError):
            save_snapshot("2026-07-09", {"x": 2}, snapshot_dir=tmp_path)  # changed → conflict

    def test_hash_is_order_independent(self):
        assert snapshot_hash({"a": 1, "b": 2}) == snapshot_hash({"b": 2, "a": 1})


class TestLineage:
    def test_hashes_stable(self):
        assert lineage_hash() == lineage_hash()
        assert len(universe_hash()) == 16

    def test_expanded_universe_differs(self):
        assert universe_hash(expanded=True) != universe_hash(expanded=False)
