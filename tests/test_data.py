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
