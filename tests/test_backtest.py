import numpy as np
import pytest

from signal_engine.backtest import BacktestResult, run_backtest
from signal_engine.config import Config


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


def test_financing_cost_reduces_returns(full_prices):
    base = run_backtest(full_prices, Config(financing_rate=0.0))
    financed = run_backtest(full_prices, Config(financing_rate=0.02, financing_threshold=0.0))
    # Charging financing on all gross notional should reduce net returns.
    assert financed.daily_returns.sum() < base.daily_returns.sum()
    # Gross returns should be unaffected by financing.
    assert financed.gross_returns.sum() == pytest.approx(base.gross_returns.sum(), rel=1e-6)
    # Financing cost series should be strictly positive on days with exposure.
    assert (financed.financing_cost >= 0).all()
    assert financed.financing_cost.sum() > 0


def test_max_annual_financing_cost_scales_positions(full_prices):
    loose = run_backtest(full_prices, Config(financing_rate=0.02, financing_threshold=0.0))
    tight = run_backtest(
        full_prices,
        Config(financing_rate=0.02, financing_threshold=0.0, max_annual_financing_cost=0.005),
    )
    # Tight financing-cost cap should reduce gross exposure and financing cost.
    assert tight.gross_exposure.mean() <= loose.gross_exposure.mean() + 1e-9
    assert tight.financing_cost.sum() < loose.financing_cost.sum()
    # It should also reduce net returns (less risk taken).
    assert tight.daily_returns.abs().mean() <= loose.daily_returns.abs().mean() + 1e-9


def test_per_instrument_cost_scheme_reduces_net_more(result, full_prices):
    flat = run_backtest(full_prices, Config(cost_scheme="flat"))
    inst = run_backtest(full_prices, Config(cost_scheme="instrument"))
    # Gross returns are identical; the instrument-specific cost scheme must not
    # produce higher net returns than the flat scheme on average (it charges more
    # for the less-liquid names in the universe).
    assert inst.gross_returns.sum() >= inst.daily_returns.sum()
    assert flat.gross_returns.sum() >= flat.daily_returns.sum()
    # Net costs: flat total costs should be <= instrument-scheme total costs.
    flat_cost = (flat.gross_returns - flat.daily_returns).sum()
    inst_cost = (inst.gross_returns - inst.daily_returns).sum()
    assert inst_cost >= flat_cost - 1e-6


def test_first_entry_trade_is_charged_cost(full_prices):
    """The initial position build-up must appear in turnover, not be dropped."""
    cfg = Config(buffer_fraction=0.0)  # disable buffer so we can observe raw entry
    res = run_backtest(full_prices, cfg)
    # Find the first day where positions move from zero (after the warm-up) to
    # non-zero.  That transition is the initial entry and should carry cost.
    notional = res.notional
    first_trade_idx = None
    for i in range(1, len(notional)):
        if notional.iloc[i].abs().sum() > 0 and notional.iloc[i - 1].abs().sum() == 0:
            first_trade_idx = i
            break
    assert first_trade_idx is not None
    assert res.turnover.iloc[first_trade_idx] > 0.0
    # Without the fix the diff() from the first NaN row produced NaN cost.
    assert np.isfinite(res.turnover.iloc[first_trade_idx])


def test_expanding_calibration_uses_no_future_data(full_prices):
    """With a calibration horizon beyond the sample, parameters stay neutral."""
    cfg = Config(calibration_min_obs=len(full_prices) + 1)
    res = run_backtest(full_prices, cfg)
    assert res.idm == 1.0
    assert res.fdm == 1.0
    n = len(res.weights)
    assert abs(sum(res.weights.values()) - 1.0) < 1e-9
    # Equal weights when calibration never triggers.
    assert max(abs(v - 1.0 / n) for v in res.weights.values()) < 1e-9


def test_default_run_uses_expanding_calibration(full_prices):
    """The default run_backtest should estimate final IDM/FDM from history only."""
    res = run_backtest(full_prices)
    # On the synthetic low-correlation panel diversification is meaningful.
    assert res.idm >= 1.0
    assert res.fdm >= 1.0
    assert abs(sum(res.weights.values()) - 1.0) < 1e-6
