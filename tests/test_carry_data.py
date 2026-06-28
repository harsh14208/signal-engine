from unittest.mock import patch

import numpy as np
import pandas as pd

from signal_engine.carry_data import build_carry_panel
from signal_engine.config import Config


def test_build_carry_panel_shape_and_aligned(small_prices):
    # small_prices uses symbols from the core universe, so some columns are bonds/equities.
    panel = build_carry_panel(small_prices)
    assert panel.shape == small_prices.shape
    assert list(panel.columns) == list(small_prices.columns)
    assert panel.index.equals(small_prices.index)


def test_build_carry_panel_zeros_for_unknown_symbols():
    idx = pd.bdate_range("2015-01-01", periods=100)
    prices = pd.DataFrame({"XXX": 100.0, "YYY": 200.0}, index=idx)
    panel = build_carry_panel(prices)
    assert (panel == 0.0).all().all()


def test_bond_carry_is_nonzero_for_bonds():
    # TLT, IEF, TIP are bond_slope instruments in the core universe.
    idx = pd.bdate_range("2020-01-01", periods=200)
    prices = pd.DataFrame({"TLT": 150.0, "GLD": 180.0}, index=idx)
    panel = build_carry_panel(prices)
    assert (panel["TLT"].dropna() != 0.0).any()


def test_expanded_universe_bond_carry():
    """SHY is an expanded-universe bond and should receive the bond carry proxy."""
    idx = pd.bdate_range("2020-01-01", periods=100)
    prices = pd.DataFrame({"SHY": 85.0, "GLD": 150.0}, index=idx)
    with patch("signal_engine.carry_data.load_bond_carry") as mock_load:
        mock_load.return_value = pd.Series(0.02, index=idx)
        panel = build_carry_panel(prices, Config(use_expanded_universe=True))
    assert (panel["SHY"] == 0.02).all()
    assert (panel["GLD"] == 0.0).all()


def test_expanded_universe_equity_carry():
    """FXI is an expanded-universe equity and should receive equity carry proxy."""
    idx = pd.bdate_range("2020-01-01", periods=100)
    prices = pd.DataFrame({"FXI": 40.0, "GLD": 150.0}, index=idx)
    with patch("signal_engine.carry_data.load_equity_carry") as mock_load:
        mock_load.return_value = pd.DataFrame({"FXI": 0.03}, index=idx)
        panel = build_carry_panel(prices, Config(use_expanded_universe=True))
    assert np.allclose(panel["FXI"], 0.03)
    assert (panel["GLD"] == 0.0).all()
