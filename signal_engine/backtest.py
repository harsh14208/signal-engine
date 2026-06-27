"""The portfolio backtest engine.

Pipeline (no lookahead — positions decided at close t-1 earn day-t P&L):

    prices ─▶ returns ─▶ blended vol ─▶ rule forecasts ─▶ combine (FDM)
           ─▶ vol-target sizing (IDM) ─▶ buffer ─▶ shift(1) ─▶ P&L − costs
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import Config
from .forecast import combine_instrument, equal_weights, estimate_fdm, pooled_rule_correlation
from .markets import BY_SYMBOL
from .portfolio import apply_buffer, estimate_idm, position_units
from .rules import carry_forecast, trend_forecasts
from .volatility import annualise, blended_daily_vol, daily_returns


@dataclass
class BacktestResult:
    daily_returns: pd.Series  # net portfolio daily returns
    gross_returns: pd.Series  # before costs
    equity: pd.Series  # net equity curve (starts at 1.0)
    per_instrument_returns: pd.DataFrame
    positions: pd.DataFrame  # effective units held
    notional: pd.DataFrame
    forecasts: pd.DataFrame  # combined forecast per instrument
    turnover: pd.Series  # daily traded-notional / capital
    instrument_corr: pd.DataFrame
    idm: float
    fdm: float
    config: Config


def _multiplier(sym: str) -> float:
    inst = BY_SYMBOL.get(sym)
    return inst.multiplier if inst else 1.0


def run_backtest(
    prices: pd.DataFrame,
    config: Config | None = None,
    carry: pd.DataFrame | None = None,
) -> BacktestResult:
    config = config or Config()
    prices = prices.sort_index().dropna(how="all").copy()
    symbols = list(prices.columns)
    returns = daily_returns(prices)

    # 1) per-instrument volatility + rule forecasts
    per_inst: dict[str, dict[str, pd.Series]] = {}
    annual_vol: dict[str, pd.Series] = {}
    for sym in symbols:
        dvol = blended_daily_vol(returns[sym])
        annual_vol[sym] = annualise(dvol)
        rf = trend_forecasts(
            prices[sym],
            dvol,
            config.ewmac_speeds,
            config.breakout_spans if config.use_breakout else (),
        )
        if config.use_carry and carry is not None and sym in carry.columns:
            rf["carry"] = carry_forecast(carry[sym], annual_vol[sym])
        per_inst[sym] = rf

    # 2) FDM from pooled rule correlation
    rule_corr = pooled_rule_correlation(per_inst)
    all_rules = sorted({r for d in per_inst.values() for r in d})
    fdm = estimate_fdm(rule_corr, equal_weights(all_rules), config.fdm_cap)

    # 3) combined forecast per instrument
    combined = {
        sym: combine_instrument(rf, equal_weights(rf.keys()), fdm, config.forecast_cap)
        for sym, rf in per_inst.items()
    }
    forecasts = pd.DataFrame(combined)

    # 4) instrument weights + IDM
    inst_w = equal_weights(symbols)
    idm = estimate_idm(returns, inst_w, config.idm_cap)

    # 5) vol-target sizing + no-trade buffer
    units = {}
    for sym in symbols:
        u = position_units(
            forecasts[sym],
            prices[sym],
            annual_vol[sym],
            config.capital,
            config.vol_target,
            inst_w[sym],
            idm,
            _multiplier(sym),
        )
        units[sym] = apply_buffer(u, config.buffer_fraction)
    units_df = pd.DataFrame(units)

    # 6) P&L (position decided at t-1 earns day-t price change) minus costs
    eff = units_df.shift(1)
    mult = pd.Series({s: _multiplier(s) for s in symbols})
    price_change = prices.diff()

    pnl = eff.mul(price_change).mul(mult, axis=1)
    traded = eff.diff().abs().mul(prices).mul(mult, axis=1)
    cost = traded * (config.cost_bps / 1e4)

    per_inst_ret = ((pnl - cost) / config.capital).fillna(0.0)
    daily_ret = per_inst_ret.sum(axis=1)
    gross_ret = (pnl / config.capital).sum(axis=1).fillna(0.0)
    notional = eff.mul(prices).mul(mult, axis=1)
    turnover = (traded.sum(axis=1) / config.capital).fillna(0.0)
    equity = (1.0 + daily_ret).cumprod()

    return BacktestResult(
        daily_returns=daily_ret,
        gross_returns=gross_ret,
        equity=equity,
        per_instrument_returns=per_inst_ret,
        positions=eff,
        notional=notional,
        forecasts=forecasts,
        turnover=turnover,
        instrument_corr=returns.corr(),
        idm=idm,
        fdm=fdm,
        config=config,
    )
