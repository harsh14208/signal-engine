import json

import numpy as np
import pandas as pd

from signal_engine import cot_data
from signal_engine.config import FORECAST_CAP


def _signal(n=2000, seed=0):
    idx = pd.bdate_range("2010-01-01", periods=n)
    walk = np.cumsum(np.random.default_rng(seed).normal(0, 0.01, n))
    return pd.Series(np.tanh(walk), idx)  # net positioning in (-1, 1)


def test_cot_forecast_capped():
    f = cot_data.cot_forecast(_signal())
    assert f.dropna().abs().max() <= FORECAST_CAP + 1e-9


def test_cot_forecast_contrarian_sign():
    # Specs getting monotonically more long → contrarian forecast goes negative.
    idx = pd.bdate_range("2010-01-01", periods=1500)
    s = pd.Series(np.linspace(-0.4, 0.4, 1500), idx)
    f = cot_data.cot_forecast(s)
    assert f.dropna().iloc[-1] < 0


def test_cot_forecast_no_lookahead():
    s = _signal()
    base = cot_data.cot_forecast(s)
    s2 = s.copy()
    s2.iloc[1000] = s2.iloc[1000] + 0.5  # shock a mid-series value
    mod = cot_data.cot_forecast(s2)
    # forecasts before the shock (minus the lag window) must be untouched
    assert np.allclose(base.iloc[:1000].to_numpy(), mod.iloc[:1000].to_numpy(), equal_nan=True)


def test_build_forecast_panel_monkeypatched(monkeypatch):
    idx = pd.bdate_range("2010-01-01", periods=1500)
    prices = pd.DataFrame({"SPY": 100.0, "EEM": 100.0}, index=idx)  # EEM not COT-mapped
    sig = pd.DataFrame({"SPY": _signal(1500, seed=3).to_numpy()}, index=idx)
    monkeypatch.setattr(
        cot_data,
        "build_cot_signal_panel",
        lambda prices, expanded=False, tag=None, refresh=False: sig.reindex(prices.index),
    )
    panel = cot_data.build_cot_forecast_panel(prices)
    assert "SPY" in panel.columns and "EEM" not in panel.columns
    assert panel.index.equals(prices.index)
    assert panel["SPY"].dropna().abs().max() <= FORECAST_CAP + 1e-9


# ── PIT cache: stitching, revision detection, rebase ─────────────────────────
def _cot_panel(values: dict, start="2024-01-05"):
    n = len(next(iter(values.values())))
    return pd.DataFrame(values, index=pd.date_range(start, periods=n, freq="7D"))


class TestStitchCotUpdate:
    def test_past_preserved_and_new_rows_appended(self):
        """A restated fresh history must NOT rewrite cached report dates; only
        genuinely new dates append."""
        old = _cot_panel({"TIP": [0.10, 0.12, 0.14]})
        fresh = _cot_panel({"TIP": [0.10, 0.30, 0.14, 0.16, 0.18]})  # mid-date restated + 2 new
        out, report = cot_data._stitch_cot_update(old, fresh)

        assert np.allclose(out["TIP"].iloc[:3].values, [0.10, 0.12, 0.14])
        assert np.allclose(out["TIP"].iloc[3:].values, [0.16, 0.18])
        assert report["n_symbols_revised"] == 1
        assert "TIP" in report["revised"]
        assert report["revised"]["TIP"]["n_dates_revised"] == 1

    def test_unrevised_symbol_clean(self):
        old = _cot_panel({"SLV": [0.05, 0.06]})
        fresh = _cot_panel({"SLV": [0.05, 0.06, 0.07]})
        out, report = cot_data._stitch_cot_update(old, fresh)
        assert report["n_symbols_revised"] == 0
        assert np.allclose(out["SLV"].values, [0.05, 0.06, 0.07])

    def test_new_symbol_taken_as_is(self):
        old = _cot_panel({"SPY": [0.01, 0.02]})
        fresh = _cot_panel({"SPY": [0.01, 0.02, 0.03], "GLD": [-0.1, -0.2, -0.3]})
        out, _ = cot_data._stitch_cot_update(old, fresh)
        assert np.allclose(out["GLD"].dropna().values, [-0.1, -0.2, -0.3])

    def test_missing_fresh_symbol_keeps_history(self):
        old = _cot_panel({"SPY": [0.01, 0.02], "GLD": [-0.1, -0.2]})
        fresh = _cot_panel({"SPY": [0.01, 0.02, 0.03]})
        out, _ = cot_data._stitch_cot_update(old, fresh)
        assert np.allclose(out["GLD"].dropna().values, [-0.1, -0.2])

    def test_total_fetch_failure_keeps_entire_cache(self):
        """An empty fresh fetch (e.g. CFTC unreachable) must not drop cached history."""
        old = _cot_panel({"SPY": [0.01, 0.02], "GLD": [-0.1, -0.2]})
        out, report = cot_data._stitch_cot_update(old, pd.DataFrame())
        assert report["n_symbols_revised"] == 0
        assert np.allclose(out["SPY"].values, [0.01, 0.02])
        assert np.allclose(out["GLD"].values, [-0.1, -0.2])


class TestBuildCotSignalPanelPIT:
    def _setup(self, tmp_path, monkeypatch, old: pd.DataFrame, fresh_by_sym: dict):
        monkeypatch.setattr(cot_data, "_CACHE_DIR", str(tmp_path))
        monkeypatch.setattr(cot_data, "COT_REVISIONS_LOG", str(tmp_path / "cot_revisions.jsonl"))
        old.to_parquet(tmp_path / "cot_signal_core.parquet")
        inc_to_sym = {tuple(v[0]): k for k, v in cot_data.COT_MAP.items()}

        def fake_fetch(inc, _exc):
            return fresh_by_sym.get(inc_to_sym.get(tuple(inc)))

        monkeypatch.setattr(cot_data, "_fetch_market", fake_fetch)

    def test_refresh_does_not_rewrite_cached_report_dates(self, tmp_path, monkeypatch):
        prices = pd.DataFrame({"SPY": 100.0}, index=pd.bdate_range("2010-01-01", periods=1600))
        old = _cot_panel({"SPY": [0.10, 0.12, 0.14]})
        fresh = pd.Series([0.10, 0.30, 0.14, 0.16], index=pd.date_range("2024-01-05", periods=4, freq="7D"))
        self._setup(tmp_path, monkeypatch, old, {"SPY": fresh})

        cot_data.build_cot_signal_panel(prices, tag="core", refresh=True)

        cached = pd.read_parquet(tmp_path / "cot_signal_core.parquet")
        assert np.allclose(cached["SPY"].iloc[:3].values, [0.10, 0.12, 0.14])  # unchanged
        assert np.allclose(cached["SPY"].iloc[3:].values, [0.16])  # new date appended

        log = (tmp_path / "cot_revisions.jsonl").read_text().strip().splitlines()
        assert len(log) == 1
        event = json.loads(log[0])
        assert event["action"] == "stitched" and "SPY" in event["revised"]

    def test_rebase_accepts_fresh_values(self, tmp_path, monkeypatch):
        prices = pd.DataFrame({"SPY": 100.0}, index=pd.bdate_range("2010-01-01", periods=1600))
        old = _cot_panel({"SPY": [0.10, 0.12]})
        fresh = pd.Series([0.30, 0.32], index=old.index)
        self._setup(tmp_path, monkeypatch, old, {"SPY": fresh})

        cot_data.build_cot_signal_panel(prices, tag="core", refresh=True, rebase=True)

        cached = pd.read_parquet(tmp_path / "cot_signal_core.parquet")
        assert np.allclose(cached["SPY"].values, [0.30, 0.32])
        log = (tmp_path / "cot_revisions.jsonl").read_text().strip().splitlines()
        assert json.loads(log[0])["action"] == "rebase"

    def test_total_fetch_failure_preserves_and_returns_cache(self, tmp_path, monkeypatch):
        """If CFTC is unreachable for every symbol, the cached history must still
        be returned rather than silently dropping the COT forecast for the day."""
        prices = pd.DataFrame({"SPY": 100.0}, index=pd.bdate_range("2024-01-01", periods=200))
        old = _cot_panel({"SPY": [0.10, 0.12]})
        self._setup(tmp_path, monkeypatch, old, {})

        panel = cot_data.build_cot_signal_panel(prices, tag="core", refresh=True)
        assert not panel.empty
        cached = pd.read_parquet(tmp_path / "cot_signal_core.parquet")
        assert np.allclose(cached["SPY"].values, [0.10, 0.12])
