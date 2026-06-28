import pandas as pd

from signal_engine.curve_data import load_2s10s_steepener, load_curve_instruments


def test_2s10s_steepener_is_positive_price_index():
    s = load_2s10s_steepener("2015-01-01", "2015-12-31")
    assert s.name == "UST2S10S"
    assert (s > 0).all()


def test_curve_instruments_align_to_prices():
    idx = pd.bdate_range("2015-01-01", periods=100)
    prices = pd.DataFrame({"SPY": 100.0}, index=idx)
    curve = load_curve_instruments(prices, steepener=True)
    assert "UST2S10S" in curve.columns
    assert curve.index.equals(prices.index)
    assert curve["UST2S10S"].notna().all()
