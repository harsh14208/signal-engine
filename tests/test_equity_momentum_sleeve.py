from unittest.mock import patch

import numpy as np
import pandas as pd

from signal_engine.equity_momentum_sleeve import build_equity_momentum_sleeve


def test_equity_momentum_sleeve_builds_price_index():
    idx = pd.bdate_range("2015-01-01", periods=400)
    # Two groups of stocks: strong and weak momentum.
    strong = 100 * (1 + np.random.normal(0.001, 0.01, len(idx))).cumprod()
    weak = 100 * (1 + np.random.normal(-0.001, 0.01, len(idx))).cumprod()

    def _mock_membership(start, end):
        return pd.DataFrame(True, index=idx, columns=["STRONG", "WEAK"])

    def _mock_close(ticker, start, end):
        if ticker == "STRONG":
            return pd.Series(strong, index=idx, name="Close")
        if ticker == "WEAK":
            return pd.Series(weak, index=idx, name="Close")
        return None

    with patch("signal_engine.equity_momentum_sleeve._load_membership", _mock_membership):
        with patch("signal_engine.equity_momentum_sleeve._load_close", _mock_close):
            with patch(
                "signal_engine.equity_momentum_sleeve._cache_path",
                return_value="/tmp/nonexistent_sp500_xsmom.parquet",
            ):
                s = build_equity_momentum_sleeve(
                    "2015-01-01",
                    "2016-09-01",
                    lookback=60,
                    rebalance=21,
                    decile=0.5,
                )
    assert s.name == "SP500_XSMOM"
    assert (s > 0).all()
    assert s.notna().all()
