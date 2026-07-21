import numpy as np
import pytest

from signal_engine.backtest import BacktestResult, _rule_weights, run_backtest
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


def test_calibration_smooth_reduces_turnover(full_prices):
    abrupt = run_backtest(full_prices, Config(calibration_smooth=None))
    smooth = run_backtest(full_prices, Config(calibration_smooth=20))
    # Smoothing should not change gross exposure much but should cut turnover.
    assert smooth.turnover.sum() <= abrupt.turnover.sum() + 1e-9
    # Realised vol should stay in a similar band.
    assert abs(smooth.daily_returns.std() * 16 - abrupt.daily_returns.std() * 16) < 0.05


def test_drawdown_control_reduces_exposure_in_drawdown(full_prices):
    base = run_backtest(full_prices, Config(use_governor=True))
    dd = run_backtest(
        full_prices,
        Config(
            use_governor=True,
            use_drawdown_control=True,
            drawdown_threshold=0.05,
            drawdown_scale=0.50,
            drawdown_recovery=0.02,
        ),
    )
    # Drawdown control should reduce average gross exposure and max drawdown.
    assert dd.gross_exposure.mean() <= base.gross_exposure.mean() + 1e-9
    assert dd.equity.min() >= base.equity.min() - 1e-6


def test_trend_strength_filter_reduces_exposure_when_weak(full_prices):
    base = run_backtest(full_prices, Config(use_governor=True))
    ts = run_backtest(
        full_prices,
        Config(
            use_governor=True,
            use_trend_strength_filter=True,
            trend_strength_window=20,
            trend_strength_threshold=0.25,
            trend_strength_scale=0.50,
        ),
    )
    # Trend-strength filter should reduce average gross exposure.
    assert ts.gross_exposure.mean() <= base.gross_exposure.mean() + 1e-9
    # Realised vol should not blow up.
    assert ts.daily_returns.std() * 16 < 0.50


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


class TestRuleWeights:
    def test_no_config_weights_is_equal(self):
        w = _rule_weights(["a", "b", "c"], Config())
        assert w == pytest.approx({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})

    def test_full_match_is_renormalised(self):
        cfg = Config(rule_weights={"a": 1.0, "b": 3.0})
        w = _rule_weights(["a", "b"], cfg)
        assert w == pytest.approx({"a": 0.25, "b": 0.75})

    def test_partial_weights_fill_missing_keys_equally_then_renormalise(self):
        # Only "a" specified; "b" gets the equal-weight default (1/2) before renorm.
        cfg = Config(rule_weights={"a": 1.0})
        w = _rule_weights(["a", "b"], cfg)
        assert w == pytest.approx({"a": 1.0 / 1.5, "b": 0.5 / 1.5})

    def test_explicit_all_zero_weights_fall_back_to_equal_not_crash(self):
        # Every active key explicitly zeroed sums to 0 — can't renormalise to 1,
        # so this falls back to equal weights rather than dividing by zero.
        cfg = Config(rule_weights={"a": 0.0, "b": 0.0})
        w = _rule_weights(["a", "b"], cfg)
        assert w == pytest.approx({"a": 0.5, "b": 0.5})
