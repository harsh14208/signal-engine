import numpy as np

from signal_engine.backtest import BacktestResult


def test_returns_result_type(result):
    assert isinstance(result, BacktestResult)
    assert result.daily_returns.notna().any()


def test_no_lookahead_first_position_is_shifted(result):
    # Positions are decided at close t-1 (shift(1)) → first row must be NaN.
    assert result.positions.iloc[0].isna().all()


def test_columns_align(result, full_prices):
    assert list(result.per_instrument_returns.columns) == list(full_prices.columns)
    assert list(result.forecasts.columns) == list(full_prices.columns)


def test_vol_targeting_in_ballpark(result):
    realised = result.daily_returns.std() * 16
    # 20% target; sizing is ex-ante so allow a wide-but-meaningful band.
    assert 0.08 < realised < 0.40, f"realised ann vol {realised:.2%} far from 20% target"


def test_equity_finite_and_positive(result):
    eq = result.equity.dropna()
    assert np.isfinite(eq).all()
    assert (eq > 0).all()


def test_costs_reduce_returns(result):
    assert result.gross_returns.sum() >= result.daily_returns.sum()
