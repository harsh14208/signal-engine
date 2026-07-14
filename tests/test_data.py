import numpy as np
import pandas as pd
import pytest

from signal_engine import data as data_module
from signal_engine.data import random_walk_panel, synthetic_carry, synthetic_prices
from signal_engine.markets import symbols


def test_synthetic_prices_shape_and_positive():
    syms = symbols()[:5]
    px = synthetic_prices(syms, n_days=500, seed=1)
    assert list(px.columns) == syms
    assert len(px) == 500
    assert (px > 0).all().all()


def test_synthetic_prices_deterministic():
    a = synthetic_prices(["SPY", "GLD"], n_days=300, seed=42)
    b = synthetic_prices(["SPY", "GLD"], n_days=300, seed=42)
    assert np.allclose(a.values, b.values)


def test_random_walk_has_near_zero_drift():
    px = random_walk_panel(8, 4000, seed=2)
    ann_drift = px.pct_change().mean() * 256
    # Driftless by construction → mean annual return hugs zero.
    assert ann_drift.abs().mean() < 0.08


def test_synthetic_carry_shape():
    px = synthetic_prices(["TLT", "IEF"], n_days=200, seed=1)
    c = synthetic_carry(["TLT", "IEF"], px.index)
    assert c.shape == (200, 2)


def test_cache_missing_symbols_raises(tmp_path, monkeypatch):
    """source='cache' must fail loudly if requested symbols are absent."""
    monkeypatch.setattr(data_module, "_CACHE_DIR", str(tmp_path))
    cache = tmp_path / "prices_test.parquet"
    pd.DataFrame({"A": [1.0, 2.0]}, index=pd.bdate_range("2020-01-01", periods=2)).to_parquet(cache)
    with pytest.raises(FileNotFoundError, match="missing requested symbols"):
        data_module.load_prices(["A", "B"], source="cache", cache_tag="test")


def test_single_symbol_yfinance_rename(monkeypatch):
    """A single-symbol yfinance download with a plain 'Close' column must be
    renamed to the requested symbol.
    """
    raw = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Adj Close": [101.0, 102.0],
            "Volume": [1e6, 1e6],
        },
        index=pd.bdate_range("2020-01-01", periods=2),
    )

    def mock_download(*_args, **_kwargs):
        return raw

    monkeypatch.setattr("yfinance.download", mock_download)
    px = data_module._fetch_yfinance(["SPY"], start="2020-01-01", end="2020-01-05")
    assert list(px.columns) == ["SPY"]
    assert (px["SPY"] == raw["Close"]).all()


# ── PIT cache: stitching, revision detection, rebase ─────────────────────────
def _panel(values: dict, start="2020-01-01"):
    n = len(next(iter(values.values())))
    return pd.DataFrame(values, index=pd.bdate_range(start, periods=n))


class TestStitchUpdate:
    def test_past_preserved_and_new_rows_appended(self):
        """A revised fresh history must NOT rewrite cached dates; new dates append
        with fresh forward returns, ratio-stitched onto the cached basis."""
        old = _panel({"TIP": [100.0, 101.0, 102.0]})
        # Upstream re-adjusted the past (×0.99) and added two new dates.
        fresh = _panel({"TIP": [99.0, 99.99, 100.98, 102.0, 103.0]})
        out, report = data_module._stitch_update(old, fresh)

        # Cached history verbatim.
        assert np.allclose(out["TIP"].iloc[:3].values, [100.0, 101.0, 102.0])
        # New rows: same forward RETURNS as fresh, on the old basis.
        expected = [102.0 * 102.0 / 100.98, 102.0 * 103.0 / 100.98]
        assert np.allclose(out["TIP"].iloc[3:].values, expected)
        # The rejected upstream revision is reported.
        assert report["n_symbols_revised"] == 1
        assert "TIP" in report["revised"]
        assert report["revised"]["TIP"]["n_dates_revised"] == 3

    def test_unrevised_symbol_clean(self):
        old = _panel({"SLV": [50.0, 51.0]})
        fresh = _panel({"SLV": [50.0, 51.0, 52.0]})
        out, report = data_module._stitch_update(old, fresh)
        assert report["n_symbols_revised"] == 0
        assert np.allclose(out["SLV"].values, [50.0, 51.0, 52.0])

    def test_new_symbol_taken_as_is(self):
        old = _panel({"SPY": [400.0, 401.0]})
        fresh = _panel({"SPY": [400.0, 401.0, 402.0], "GLD": [180.0, 181.0, 182.0]})
        out, _ = data_module._stitch_update(old, fresh)
        assert np.allclose(out["GLD"].dropna().values, [180.0, 181.0, 182.0])

    def test_missing_fresh_symbol_keeps_history(self):
        """A failed/rate-limited fetch for one symbol must not drop its history."""
        old = _panel({"SPY": [400.0, 401.0], "GLD": [180.0, 181.0]})
        fresh = _panel({"SPY": [400.0, 401.0, 402.0]})
        out, _ = data_module._stitch_update(old, fresh)
        assert np.allclose(out["GLD"].dropna().values, [180.0, 181.0])


class TestLoadPricesPIT:
    def _setup(self, tmp_path, monkeypatch, old, fresh):
        monkeypatch.setattr(data_module, "_CACHE_DIR", str(tmp_path))
        monkeypatch.setattr(data_module, "REVISIONS_LOG", str(tmp_path / "price_revisions.jsonl"))
        old.to_parquet(tmp_path / "prices_pit.parquet")
        monkeypatch.setattr(data_module, "_fetch_yfinance", lambda *a, **k: fresh)

    def test_yfinance_refresh_does_not_rewrite_history(self, tmp_path, monkeypatch):
        n = 400  # > 300-bar _clean filter
        base = np.linspace(100.0, 140.0, n)
        old = _panel({"TIP": base.tolist()})
        fresh_hist = (base * 0.99).tolist() + [141.0, 142.0]
        fresh = _panel({"TIP": fresh_hist})
        self._setup(tmp_path, monkeypatch, old, fresh)

        px = data_module.load_prices(["TIP"], source="yfinance", cache_tag="pit")
        assert np.allclose(px["TIP"].iloc[:n].values, base)  # past immutable
        assert len(px) == n + 2
        # Revision logged.
        log = (tmp_path / "price_revisions.jsonl").read_text().strip().splitlines()
        assert len(log) == 1
        import json as _json

        event = _json.loads(log[0])
        assert event["action"] == "stitched" and "TIP" in event["revised"]

    def test_rebase_accepts_fresh_history(self, tmp_path, monkeypatch):
        n = 400
        base = np.linspace(100.0, 140.0, n)
        old = _panel({"TIP": base.tolist()})
        fresh = _panel({"TIP": (base * 0.99).tolist()})
        self._setup(tmp_path, monkeypatch, old, fresh)

        px = data_module.load_prices(["TIP"], source="yfinance", cache_tag="pit", rebase=True)
        assert np.allclose(px["TIP"].values, base * 0.99)  # fresh wholesale
        log = (tmp_path / "price_revisions.jsonl").read_text().strip().splitlines()
        import json as _json

        assert _json.loads(log[0])["action"] == "rebase"
