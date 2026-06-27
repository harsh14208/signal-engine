"""The portfolio backtest engine.

Pipeline (no lookahead — positions decided at close t-1 earn day-t P&L):

    prices ─▶ returns ─▶ blended vol ─▶ rule forecasts ─▶ combine (FDM)
           ─▶ cluster-weighted vol-target sizing (IDM) ─▶ realised-vol governor
           ─▶ no-trade buffer ─▶ shift(1) ─▶ P&L − costs

The governor is a two-pass overlay: pass 1 simulates the raw (ungoverned) book
to estimate its realised vol; pass 2 scales every position by a lagged
target/realised multiplier so realised vol lands near target.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import Config
from .forecast import combine_instrument, equal_weights, estimate_fdm, pooled_rule_correlation
from .markets import BY_SYMBOL
from .portfolio import apply_buffer, estimate_idm, position_units, vol_governor
from .rules import carry_forecast, trend_forecasts
from .volatility import annualise, blended_daily_vol, daily_returns
from .weights import cluster_weights


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
    governor: pd.Series  # applied leverage multiplier (1.0 if disabled)
    weights: dict  # instrument risk weights actually used
    idm: float
    fdm: float
    config: Config


def _multiplier(sym: str) -> float:
    inst = BY_SYMBOL.get(sym)
    return inst.multiplier if inst else 1.0


def _simulate(
    units_unbuffered: pd.DataFrame,
    prices: pd.DataFrame,
    mult: pd.Series,
    cost_bps: float,
    capital: float,
    buffer_fraction: float,
) -> dict:
    """Buffer → shift(1) → P&L − costs for a units DataFrame."""
    buffered = pd.DataFrame(
        {c: apply_buffer(units_unbuffered[c], buffer_fraction) for c in units_unbuffered.columns}
    )
    eff = buffered.shift(1)
    price_change = prices.diff()
    pnl = eff.mul(price_change).mul(mult, axis=1)
    traded = eff.diff().abs().mul(prices).mul(mult, axis=1)
    cost = traded * (cost_bps / 1e4)
    per_inst = ((pnl - cost) / capital).fillna(0.0)
    return {
        "eff": eff,
        "per_inst": per_inst,
        "daily": per_inst.sum(axis=1),
        "gross": (pnl / capital).sum(axis=1).fillna(0.0),
        "notional": eff.mul(prices).mul(mult, axis=1),
        "turnover": (traded.sum(axis=1) / capital).fillna(0.0),
    }


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
    forecasts = pd.DataFrame(
        {
            sym: combine_instrument(rf, equal_weights(rf.keys()), fdm, config.forecast_cap)
            for sym, rf in per_inst.items()
        }
    )

    # 4) instrument risk weights (cluster handcrafting) + IDM
    weights = cluster_weights(symbols) if config.cluster_weights else equal_weights(symbols)
    idm = estimate_idm(returns, weights, config.idm_cap)

    # 5) raw vol-target sizing (pre-governor, pre-buffer)
    mult = pd.Series({s: _multiplier(s) for s in symbols})
    raw_units = pd.DataFrame(
        {
            sym: position_units(
                forecasts[sym],
                prices[sym],
                annual_vol[sym],
                config.capital,
                config.vol_target,
                weights[sym],
                idm,
                _multiplier(sym),
            )
            for sym in symbols
        }
    )

    # 6) realised-vol governor (two-pass; multiplier is lagged → no lookahead)
    if config.use_governor:
        sim_raw = _simulate(
            raw_units, prices, mult, config.cost_bps, config.capital, config.buffer_fraction
        )
        governor = vol_governor(
            sim_raw["daily"],
            config.vol_target,
            config.governor_span,
            config.governor_min,
            config.governor_max,
        )
        governed = raw_units.mul(governor, axis=0)
    else:
        governor = pd.Series(1.0, index=prices.index)
        governed = raw_units

    sim = _simulate(governed, prices, mult, config.cost_bps, config.capital, config.buffer_fraction)
    equity = (1.0 + sim["daily"]).cumprod()

    return BacktestResult(
        daily_returns=sim["daily"],
        gross_returns=sim["gross"],
        equity=equity,
        per_instrument_returns=sim["per_inst"],
        positions=sim["eff"],
        notional=sim["notional"],
        forecasts=forecasts,
        turnover=sim["turnover"],
        instrument_corr=returns.corr(),
        governor=governor,
        weights=weights,
        idm=idm,
        fdm=fdm,
        config=config,
    )
