import numpy as np
import pandas as pd

from signal_engine import vrp_data


def _vix_and_underlying(n=600, seed=0):
    idx = pd.bdate_range("2010-01-01", periods=n)
    rng = np.random.default_rng(seed)
    vix = pd.Series(15.0 + 8.0 * np.abs(rng.normal(0, 1, n)), idx)  # wandering ~15-25
    under = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)), idx)
    return vix, under


def test_short_vol_price_positive_no_nan():
    vix, under = _vix_and_underlying()
    p = vrp_data.short_vol_price(vix, under)
    tail = p.iloc[120:]
    assert (tail > 0).all()
    assert not tail.isna().any()


def test_short_vol_price_no_lookahead():
    vix, under = _vix_and_underlying()
    base = vrp_data.short_vol_price(vix, under)
    under2 = under.copy()
    under2.iloc[-1] = under2.iloc[-1] * 1.10  # shock only the last day
    mod = vrp_data.short_vol_price(vix, under2)
    assert np.allclose(base.iloc[:-1].to_numpy(), mod.iloc[:-1].to_numpy(), equal_nan=True)


def test_build_vrp_sleeve_monkeypatched(monkeypatch):
    idx = pd.bdate_range("2010-01-01", periods=400)
    prices = pd.DataFrame(
        {"SPY": 100 * np.cumprod(1 + np.random.default_rng(1).normal(0, 0.01, 400))},
        index=idx,
    )
    monkeypatch.setattr(vrp_data, "_load_yf_series", lambda sym, s, e: pd.Series(18.0, index=idx))
    sleeve = vrp_data.build_vrp_sleeve(prices, mapping={"^VIX": "SPY"})
    assert "SPY_VRP" in sleeve.columns
    assert len(sleeve) == len(prices)
    assert sleeve["SPY_VRP"].notna().iloc[150:].all()


def test_build_vrp_sleeve_skips_missing_underlying(monkeypatch):
    idx = pd.bdate_range("2010-01-01", periods=300)
    prices = pd.DataFrame({"GLD": 100.0}, index=idx)
    monkeypatch.setattr(vrp_data, "_load_yf_series", lambda sym, s, e: pd.Series(18.0, index=idx))
    # mapping points at SPY which is absent → empty sleeve
    sleeve = vrp_data.build_vrp_sleeve(prices, mapping={"^VIX": "SPY"})
    assert sleeve.empty or "SPY_VRP" not in sleeve.columns
