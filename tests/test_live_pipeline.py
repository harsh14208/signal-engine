"""Tests for the Tier A forward-deployment helpers and scripts."""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from signal_engine import live
from signal_engine.backtest import run_backtest
from signal_engine.live import (
    append_shadow_return,
    build_target_record,
    compute_shadow_return,
    config_from_target,
    generate_target,
    load_all_targets,
    load_latest_target,
    load_live_returns,
    read_kill_switch,
    run_reconciliation,
    validated_config,
)


@pytest.fixture
def cot_panel(full_prices):
    """A dummy COT forecast panel (zero forecast) aligned to full_prices."""
    return pd.DataFrame(0.0, index=full_prices.index, columns=list(full_prices.columns)[:5])


@pytest.fixture
def backtest_result(full_prices, cot_panel):
    cfg = validated_config(cot=True)
    return run_backtest(full_prices, cfg, cot=cot_panel)


class TestConfigAndRecords:
    def test_validated_config_defaults(self):
        cfg = validated_config()
        assert cfg.use_cot is True
        assert cfg.cot_momentum is False
        assert cfg.buffer_fraction == pytest.approx(0.30)
        assert cfg.use_governor is True

    def test_config_from_target_round_trip(self):
        target = {
            "capital": 500_000.0,
            "vol_target": 0.15,
            "buffer_fraction": 0.25,
            "use_cot": False,
            "cot_momentum": True,
        }
        cfg = config_from_target(target)
        assert cfg.capital == pytest.approx(500_000.0)
        assert cfg.vol_target == pytest.approx(0.15)
        assert cfg.buffer_fraction == pytest.approx(0.25)
        assert cfg.use_cot is False
        assert cfg.cot_momentum is True

    def test_build_target_record(self, backtest_result):
        cfg = validated_config(cot=True)
        record = build_target_record(backtest_result, cfg)
        assert record["date"] == backtest_result.daily_returns.index[-1].strftime("%Y-%m-%d")
        assert record["use_cot"] is True
        assert "units" in record
        assert "notional" in record
        assert "forecast" in record
        assert record["idm"] > 0
        assert record["fdm"] > 0

    def test_config_from_target_full_round_trip(self, backtest_result):
        cfg = validated_config(
            cot=False,
            use_network_momentum=True,
            use_xsmom=True,
            use_crypto=True,
            use_regime_overlay=True,
            max_gross_notional=2.5,
            financing_rate=0.01,
            ewmac_speeds=((8, 32), (16, 64)),
        )
        record = build_target_record(backtest_result, cfg, book="challenger")
        restored = config_from_target(record)
        assert restored.use_cot is False
        assert restored.use_network_momentum is True
        assert restored.use_xsmom is True
        assert restored.use_crypto is True
        assert restored.use_regime_overlay is True
        assert restored.max_gross_notional == pytest.approx(2.5)
        assert restored.financing_rate == pytest.approx(0.01)
        assert restored.ewmac_speeds == ((8, 32), (16, 64))


class TestGenerateTarget:
    def test_generate_target_rejects_insufficient_history(self, tmp_path, full_prices, cot_panel):
        targets = tmp_path / "targets.jsonl"
        # Slice to fewer rows than the config requires for a warm restart.
        short_prices = full_prices.iloc[:50]
        with patch.object(live, "load_prices", return_value=short_prices):
            with pytest.raises(RuntimeError, match="Insufficient history"):
                generate_target(
                    source="cache",
                    cot=False,
                    refresh_cot=False,
                    targets_path=targets,
                    snapshot_dir=tmp_path / "snaps",
                )

    def test_generate_target_writes_and_is_idempotent(self, tmp_path, full_prices, cot_panel):
        targets = tmp_path / "targets.jsonl"
        with patch.object(live, "load_prices", return_value=full_prices):
            with patch.object(
                live, "_build_cot_forecast_panel_with_fallback", return_value=cot_panel
            ):
                res1 = generate_target(
                    source="cache", cot=True, refresh_cot=False, targets_path=targets, snapshot_dir=tmp_path / "snaps"
                )

        assert "record" in res1
        record = res1["record"]
        assert targets.exists()
        assert load_latest_target(targets)["date"] == record["date"]

        with patch.object(live, "load_prices", return_value=full_prices):
            with patch.object(
                live, "_build_cot_forecast_panel_with_fallback", return_value=cot_panel
            ):
                res2 = generate_target(
                    source="cache", cot=True, refresh_cot=False, targets_path=targets, snapshot_dir=tmp_path / "snaps"
                )

        assert res2.get("skipped") is True
        assert len(load_all_targets(targets)) == 1

    def test_generate_target_without_cot(self, tmp_path, full_prices):
        targets = tmp_path / "targets.jsonl"
        with patch.object(live, "load_prices", return_value=full_prices):
            res = generate_target(
                source="cache", cot=False, refresh_cot=False, targets_path=targets, snapshot_dir=tmp_path / "snaps"
            )
        assert res["record"]["use_cot"] is False
        assert res["record"]["cot_as_of"] is None

    def test_generate_target_honors_end_date(self, tmp_path, full_prices):
        targets = tmp_path / "targets.jsonl"
        end_date = full_prices.index[-10].strftime("%Y-%m-%d")
        with patch.object(live, "load_prices", return_value=full_prices):
            res = generate_target(
                source="cache",
                cot=False,
                refresh_cot=False,
                end=end_date,
                targets_path=targets,
                snapshot_dir=tmp_path / "snaps",
            )
        assert res["record"]["date"] == end_date
        assert res["record"]["as_of"] == end_date


class TestShadowBook:
    def test_compute_shadow_return(self):
        idx = pd.bdate_range("2020-01-01", periods=5)
        prices = pd.DataFrame({"SPY": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=idx)
        target = {
            "date": "2020-01-02",
            "capital": 100_000.0,
            "units": {"SPY": 10.0},
        }
        mark_date, ret = compute_shadow_return(target, prices)
        assert mark_date == pd.Timestamp("2020-01-03")
        # 10 shares * $1 price change / $100k capital = 0.01%
        assert ret == pytest.approx(10.0 * 1.0 / 100_000.0)

    def test_append_shadow_return_idempotent(self, tmp_path):
        returns_path = tmp_path / "returns.csv"
        idx = pd.bdate_range("2020-01-01", periods=4)
        prices = pd.DataFrame({"SPY": [100.0, 101.0, 102.0, 103.0]}, index=idx)
        target = {
            "date": "2020-01-02",
            "capital": 100_000.0,
            "units": {"SPY": 10.0},
            "use_cot": True,
        }
        res1 = append_shadow_return(target, prices=prices, returns_path=returns_path)
        assert res1["record"]["date"] == "2020-01-03"
        assert returns_path.exists()

        res2 = append_shadow_return(target, prices=prices, returns_path=returns_path)
        assert res2.get("skipped") is True
        df = pd.read_csv(returns_path)
        assert len(df) == 1

    def test_load_live_returns(self, tmp_path):
        returns_path = tmp_path / "returns.csv"
        pd.DataFrame(
            {
                "date": ["2020-01-03", "2020-01-04"],
                "live_return": [0.001, -0.0005],
                "mode": ["shadow", "shadow"],
                "use_cot": [True, True],
            }
        ).to_csv(returns_path, index=False)
        s = load_live_returns(returns_path)
        assert len(s) == 2
        assert s.index[0] == pd.Timestamp("2020-01-03")


class TestReconciliation:
    def test_run_reconciliation_aligned(self, tmp_path, full_prices, cot_panel):
        targets = tmp_path / "targets.jsonl"
        returns_path = tmp_path / "returns.csv"
        recon_dir = tmp_path / "recon"
        kill_switch = tmp_path / "kill.json"

        cfg = validated_config(cot=True)
        result = run_backtest(full_prices, cfg, cot=cot_panel)
        record = build_target_record(result, cfg, cot=cot_panel)
        targets.write_text(json.dumps(record) + "\n")

        # Live returns = modeled GROSS returns (shadow has no costs) → perfect tracking.
        live_df = pd.DataFrame(
            {
                "date": result.gross_returns.index.strftime("%Y-%m-%d"),
                "live_return": result.gross_returns.values,
                "mode": "shadow",
                "use_cot": True,
                "book": "champion",
            }
        )
        live_df.to_csv(returns_path, index=False)

        with patch.object(live, "load_prices", return_value=full_prices):
            with patch.object(
                live, "_build_cot_forecast_panel_with_fallback", return_value=cot_panel
            ):
                res = run_reconciliation(
                    target=load_latest_target(targets),
                    source="cache",
                    returns_path=returns_path,
                    recon_dir=recon_dir,
                    kill_switch_path=kill_switch,
                    alarm_floor=-10.0,
                )

        rec = res["report"]["reconciliation"]
        assert rec["corr"] > 0.95
        assert rec["aligned"] is True
        assert res["report"]["compare_to"] == "gross"
        assert res["kill_switch"]["paused"] is False
        assert len(list(recon_dir.glob("*.json"))) == 1

    def test_run_reconciliation_triggers_kill_switch(self, tmp_path, full_prices, cot_panel):
        targets = tmp_path / "targets.jsonl"
        returns_path = tmp_path / "returns.csv"
        recon_dir = tmp_path / "recon"
        kill_switch = tmp_path / "kill.json"

        cfg = validated_config(cot=True)
        result = run_backtest(full_prices, cfg, cot=cot_panel)
        record = build_target_record(result, cfg, cot=cot_panel)
        targets.write_text(json.dumps(record) + "\n")

        # Live returns uncorrelated with modeled returns → tracking error explodes.
        np.random.seed(42)
        noise = pd.Series(
            np.random.normal(0, 0.01, len(result.daily_returns)), index=result.daily_returns.index
        )
        live_df = pd.DataFrame(
            {
                "date": noise.index.strftime("%Y-%m-%d"),
                "live_return": noise.values,
                "mode": "shadow",
                "use_cot": True,
                "book": "champion",
            }
        )
        live_df.to_csv(returns_path, index=False)

        with patch.object(live, "load_prices", return_value=full_prices):
            with patch.object(
                live, "_build_cot_forecast_panel_with_fallback", return_value=cot_panel
            ):
                res = run_reconciliation(
                    target=load_latest_target(targets),
                    source="cache",
                    returns_path=returns_path,
                    recon_dir=recon_dir,
                    kill_switch_path=kill_switch,
                )

        assert res["kill_switch"]["paused"] is True
        assert res["kill_switch"]["reason"] == "tracking_error"
        assert read_kill_switch(kill_switch)["paused"] is True


class TestAlpacaScript:
    def test_execute_alpaca_skips_when_kill_switch_engaged(self, tmp_path):
        targets = tmp_path / "targets.jsonl"
        kill_switch = tmp_path / "kill.json"
        targets.write_text(json.dumps({"date": str(date.today()), "units": {"SPY": 1.0}}) + "\n")
        kill_switch.write_text(json.dumps({"paused": True, "reason": "test"}))

        from scripts.execute_alpaca import execute_targets

        res = execute_targets(
            target=load_latest_target(targets),
            live=False,
            orders_path=tmp_path / "orders.jsonl",
            kill_switch=read_kill_switch(kill_switch),
        )
        assert res["skipped"] is True
        assert res["reason"] == "kill_switch_engaged"


class TestInputRevision:
    def test_flags_revised_symbol(self):
        from signal_engine.live import input_revision_report

        dates = pd.bdate_range("2026-07-06", periods=3)
        targets = [
            {"date": d.strftime("%Y-%m-%d"), "forecast": {"TIP": -10.4, "SLV": -13.7, "TLT": None}}
            for d in dates
        ]
        # Recomputed today: TIP flipped (data revision), SLV unchanged.
        recomputed = pd.DataFrame({"TIP": [5.3, 5.3, 5.3], "SLV": [-13.7, -13.7, -13.7]}, index=dates)
        rep = input_revision_report(targets, recomputed)
        assert rep["clean"] is False
        assert rep["n_symbols_revised"] == 1
        assert "TIP" in rep["revised"]
        assert rep["revised"]["TIP"]["max_abs_diff"] == pytest.approx(15.7)
        assert rep["n_targets_checked"] == 3

    def test_clean_when_reproducible(self):
        from signal_engine.live import input_revision_report

        dates = pd.bdate_range("2026-07-06", periods=2)
        targets = [{"date": d.strftime("%Y-%m-%d"), "forecast": {"SPY": 15.8}} for d in dates]
        recomputed = pd.DataFrame({"SPY": [15.8, 15.8]}, index=dates)
        rep = input_revision_report(targets, recomputed)
        assert rep["clean"] is True and rep["n_symbols_revised"] == 0


class TestChallengerSemis:
    """The challenger_semis book: champion config + the semis pack, forward-tested."""

    def _prices_with_semis(self, full_prices):
        from signal_engine.data import synthetic_prices

        extras = synthetic_prices(["SMH", "SOXX", "XSD"], n_days=len(full_prices), seed=9)
        extras.index = full_prices.index
        return pd.concat([full_prices, extras], axis=1)

    def test_generate_target_with_extra_symbols(self, tmp_path, full_prices):
        panel = self._prices_with_semis(full_prices)
        requested: list[list[str]] = []

        def fake_load_prices(syms, **kwargs):
            requested.append(list(syms))
            return panel

        targets = tmp_path / "targets.jsonl"
        with patch.object(live, "load_prices", side_effect=fake_load_prices):
            res = generate_target(
                source="cache",
                cot=False,
                targets_path=targets,
                book="challenger_semis",
                extra_symbols=["SMH", "SOXX", "XSD"],
                snapshot=False,
            )

        record = res["record"]
        assert record["book"] == "challenger_semis"
        assert record["universe_extra"] == ["SMH", "SOXX", "XSD"]
        assert {"SMH", "SOXX", "XSD"} <= set(record["units"])
        # The price request itself must include the pack.
        assert {"SMH", "SOXX", "XSD"} <= set(requested[0])

    def test_reconciliation_backtest_uses_target_universe(self, full_prices):
        from signal_engine.live import build_backtest_for_reconciliation

        panel = self._prices_with_semis(full_prices)
        requested: list[list[str]] = []

        def fake_load_prices(syms, **kwargs):
            requested.append(list(syms))
            return panel

        target = {
            "date": "2020-01-02",
            "use_cot": False,
            "universe_extra": ["SMH", "SOXX", "XSD"],
        }
        with patch.object(live, "load_prices", side_effect=fake_load_prices):
            _, result = build_backtest_for_reconciliation(target, source="cache")
        assert {"SMH", "SOXX", "XSD"} <= set(requested[0])
        assert "SMH" in result.forecasts.columns

    def test_executor_ignores_trailing_challenger(self, tmp_path):
        """The broker must trade the champion even when a challenger is written last —
        and refuse to run when no champion exists at all."""
        from signal_engine.live import load_latest_target_for_book

        targets = tmp_path / "targets.jsonl"
        # Champion written first, challenger appended after (as forward_loop.sh does
        # nightly) — the executor must still resolve to the champion, not the
        # trailing challenger record.
        targets.write_text(
            json.dumps({"date": "2026-07-15", "book": "champion", "units": {"SPY": 5.0}})
            + "\n"
            + json.dumps(
                {"date": "2026-07-15", "book": "challenger_semis", "units": {"SMH": 10.0}}
            )
            + "\n"
        )
        target = load_latest_target_for_book(targets, "champion")
        assert target is not None
        assert target["book"] == "champion"
        assert target["units"] == {"SPY": 5.0}

    def test_executor_refuses_without_champion(self, tmp_path):
        """No champion record at all → the executor must refuse, not trade a challenger."""
        from scripts.execute_alpaca import main as exec_main

        targets = tmp_path / "targets.jsonl"
        # Only challenger records → the executor must refuse, not trade the challenger.
        targets.write_text(
            json.dumps(
                {"date": "2026-07-15", "book": "challenger_semis", "units": {"SMH": 10.0}}
            )
            + "\n"
        )
        rc = exec_main(["--paper", "--targets", str(targets)])
        assert rc == 1  # "No champion target record found."
