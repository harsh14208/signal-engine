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
