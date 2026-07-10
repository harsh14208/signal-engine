"""Phase 3 tests — replay-based decision-level drift detection."""

from __future__ import annotations

import copy
import json
from unittest.mock import patch

import pytest

import signal_engine.live as live
from signal_engine.data import synthetic_prices
from signal_engine.feature_store import load_snapshot, save_snapshot
from signal_engine.markets import symbols
from signal_engine.replay import config_from_snapshot, detect_drift, replay_decision


@pytest.fixture(scope="module")
def prices():
    return synthetic_prices(symbols(), n_days=1200, seed=5)


@pytest.fixture()
def snapshot(prices, tmp_path):
    """Generate a real target (and its snapshot) from the synthetic panel."""
    with patch.object(live, "load_prices", return_value=prices):
        res = live.generate_target(
            source="cache", cot=False, refresh_cot=False,
            targets_path=tmp_path / "t.jsonl", snapshot_dir=tmp_path / "snaps",
        )
    return load_snapshot(res["record"]["date"], snapshot_dir=tmp_path / "snaps")


class TestReplayDecision:
    def test_faithful_replay_matches(self, snapshot, prices):
        r = replay_decision(snapshot, prices=prices, cot=None)
        assert r["kind"] == "matched"
        assert r["alarm"] is False
        assert r["units_max_diff"] == pytest.approx(0.0, abs=1e-6)

    def test_tampered_units_is_logic_drift(self, snapshot, prices):
        bad = copy.deepcopy(snapshot)
        k = next(iter(bad["features"]["units"]))
        bad["features"]["units"][k] = (bad["features"]["units"][k] or 0.0) + 25.0
        r = replay_decision(bad, prices=prices, cot=None)
        assert r["kind"] == "logic"
        assert r["alarm"] is True
        assert r["units_max_diff"] >= 25.0

    def test_revised_data_is_data_drift_not_alarm(self, snapshot, prices):
        bad = copy.deepcopy(snapshot)
        k = next(iter(bad["features"]["units"]))
        bad["features"]["units"][k] = (bad["features"]["units"][k] or 0.0) + 25.0
        bad["features"]["price_fingerprint"] = "deadbeefdeadbeef"
        r = replay_decision(bad, prices=prices, cot=None)
        assert r["kind"] == "data"
        assert r["alarm"] is False

    def test_lineage_change_is_not_alarm(self, snapshot, prices):
        bad = copy.deepcopy(snapshot)
        k = next(iter(bad["features"]["units"]))
        bad["features"]["units"][k] = (bad["features"]["units"][k] or 0.0) + 25.0
        bad["lineage_hash"] = "0000000000000000"  # different data-handling regime
        r = replay_decision(bad, prices=prices, cot=None)
        assert r["kind"] == "lineage"
        assert r["alarm"] is False


class TestConfigFromSnapshot:
    def test_rebuilds_snapshot_config(self, snapshot):
        cfg = config_from_snapshot(snapshot["features"])
        assert cfg.use_cot is False
        assert cfg.buffer_fraction == pytest.approx(snapshot["features"]["config"]["buffer_fraction"])


class TestDetectDrift:
    def test_all_matched(self, snapshot, prices, tmp_path):
        # Re-save the (untampered) snapshot into a clean dir and replay the panel.
        snap_dir = tmp_path / "clean"
        save_snapshot(snapshot["as_of"], snapshot["features"], book=snapshot["book"],
                      snapshot_dir=snap_dir)
        rep = detect_drift(snapshot_dir=snap_dir, prices=prices, cot=None)
        assert rep["n_snapshots"] == 1
        assert rep["matched"] == 1
        assert rep["alarm"] is False

    def test_logic_drift_trips_alarm(self, snapshot, prices, tmp_path):
        snap_dir = tmp_path / "drift"
        tampered = copy.deepcopy(snapshot)
        k = next(iter(tampered["features"]["units"]))
        tampered["features"]["units"][k] = (tampered["features"]["units"][k] or 0.0) + 25.0
        # Write the tampered envelope directly (bypass immutability check).
        snap_dir.mkdir(parents=True)
        path = snap_dir / f"{tampered['as_of']}_{tampered['book']}.json"
        path.write_text(json.dumps(tampered))
        rep = detect_drift(snapshot_dir=snap_dir, prices=prices, cot=None)
        assert rep["alarm"] is True
        assert rep["by_kind"].get("logic") == 1
        assert len(rep["alarms"]) == 1

    def test_empty_dir(self, tmp_path):
        rep = detect_drift(snapshot_dir=tmp_path / "none", prices=None)
        assert rep["n_snapshots"] == 0
        assert rep["alarm"] is False
