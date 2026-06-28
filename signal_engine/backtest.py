"""The portfolio backtest engine.

Pipeline (no lookahead — positions decided at close t-1 earn day-t P&L):

    prices ─▶ returns ─▶ blended vol ─▶ rule forecasts ─▶ combine (FDM)
           ─▶ instrument-weighted vol-target sizing (IDM) ─▶ corr-spike de-risk
           ─▶ regime overlay ─▶ realised-vol governor ─▶ no-trade buffer
           ─▶ shift(1) ─▶ P&L − costs

The governor is a two-pass overlay: pass 1 simulates the raw (ungoverned) book
with the correlation-spike and regime overlays applied to estimate its realised
vol; pass 2 scales every position by a lagged target/realised multiplier so
realised vol lands near target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .forecast import (
    combine_instrument,
    equal_weights as _equal_rule_weights,
    estimate_fdm,
    pooled_rule_correlation,
)
from .markets import cost_per_symbol, instrument_for
from .portfolio import apply_buffer, corr_spike_overlay, estimate_idm, position_units, vol_governor
from .rules import (
    acceleration_forecast,
    carry_forecast,
    cross_sectional_momentum_forecast,
    trend_forecasts,
)
from .scalars import _dynamic_scalar
from .volatility import annualise, blended_daily_vol, daily_returns
from .weights import build_instrument_weights


@dataclass
class BacktestResult:
    daily_returns: pd.Series  # net portfolio daily returns
    gross_returns: pd.Series  # before costs
    equity: pd.Series  # net equity curve (starts at 1.0)
    per_instrument_returns: pd.DataFrame
    per_instrument_gross: pd.DataFrame
    positions: pd.DataFrame  # effective units held (shifted buffered targets)
    buffered: pd.DataFrame  # post-buffer target units at each close
    notional: pd.DataFrame
    forecasts: pd.DataFrame  # combined forecast per instrument
    turnover: pd.Series  # daily traded-notional / capital
    instrument_corr: pd.DataFrame
    governor: pd.Series  # applied leverage multiplier (1.0 if disabled)
    overlay: pd.Series  # correlation-spike de-gross multiplier (1.0 if disabled)
    regime: pd.Series  # macro regime de-gross multiplier (1.0 if disabled)
    weights: dict  # instrument risk weights actually used
    idm: float
    fdm: float
    config: Config


def _multiplier(sym: str, expanded: bool = False) -> float:
    inst = instrument_for(sym, expanded)
    return inst.multiplier if inst else 1.0


def _cost_per_symbol(symbols: list[str], config: Config) -> pd.Series:
    """Return a Series of per-side bps costs for each symbol."""
    if config.cost_scheme == "instrument":
        mapping = cost_per_symbol(config.use_expanded_universe)
        return pd.Series({s: mapping.get(s, config.cost_bps) for s in symbols})
    return pd.Series({s: float(config.cost_bps) for s in symbols})


def _rule_weights(keys: list[str], config: Config) -> dict[str, float]:
    """Use Config.rule_weights if supplied, otherwise equal weights."""
    keys = list(keys)
    if not keys:
        return {}
    if config.rule_weights and all(k in config.rule_weights for k in keys):
        total = sum(config.rule_weights[k] for k in keys)
        return {k: config.rule_weights[k] / total for k in keys}
    return _equal_rule_weights(keys)


def _simulate(
    units_unbuffered: pd.DataFrame,
    prices: pd.DataFrame,
    mult: pd.Series,
    cost_bps: pd.Series,
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
    # Charge the initial entry as a trade (eff.diff() starts from NaN and drops it).
    traded = traded.fillna(eff.abs().mul(prices).mul(mult, axis=1))
    cost = traded.mul(cost_bps, axis=1) / 1e4
    per_inst_gross = (pnl / capital).fillna(0.0)
    per_inst = ((pnl - cost) / capital).fillna(0.0)
    return {
        "eff": eff,
        "per_inst": per_inst,
        "per_inst_gross": per_inst_gross,
        "daily": per_inst.sum(axis=1),
        "gross": (pnl / capital).sum(axis=1).fillna(0.0),
        "notional": eff.mul(prices).mul(mult, axis=1),
        "turnover": (traded.sum(axis=1) / capital).fillna(0.0),
    }


def _build_forecasts(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    config: Config,
    carry: pd.DataFrame | None,
    cot: pd.DataFrame | None = None,
) -> tuple[dict[str, dict[str, pd.Series]], dict[str, pd.Series]]:
    """Compute per-instrument rule forecasts and annualised vols."""
    symbols = list(prices.columns)
    per_inst: dict[str, dict[str, pd.Series]] = {}
    annual_vol: dict[str, pd.Series] = {}

    for sym in symbols:
        dvol = blended_daily_vol(
            returns[sym],
            use_garch=config.use_garch_vol,
            garch_weight=config.garch_weight,
            garch_horizon=config.garch_horizon,
            garch_min_history=config.garch_min_history,
            garch_refit_step=config.garch_refit_step,
        )
        annual_vol[sym] = annualise(dvol)
        rf = trend_forecasts(
            prices[sym],
            dvol,
            config.ewmac_speeds,
            config.breakout_spans if config.use_breakout else (),
        )
        if (
            (config.use_carry or config.use_carry_proxies)
            and carry is not None
            and sym in carry.columns
        ):
            rf["carry"] = carry_forecast(carry[sym], annual_vol[sym])
        if config.use_cot and cot is not None and sym in cot.columns:
            # Pre-computed full-history forecast (causal); reindex to this slice.
            rf["cot"] = cot[sym].reindex(prices.index)
        if config.use_accel:
            fast, slow = config.accel_speeds[0], config.accel_speeds[1]
            rf["accel"] = acceleration_forecast(prices[sym], dvol, fast_pair=fast, slow_pair=slow)

        # Optional expanding-window scalar recalibration.
        if config.use_empirical_scalars:
            rf = {
                name: (fc * _dynamic_scalar(fc)).clip(-config.forecast_cap, config.forecast_cap)
                for name, fc in rf.items()
            }

        per_inst[sym] = rf

    if config.use_xsmom:
        xsmom = cross_sectional_momentum_forecast(prices, lookback=config.xsmom_lookback)
        for sym in symbols:
            if sym in xsmom.columns:
                per_inst[sym]["xsmom"] = xsmom[sym]

    return per_inst, annual_vol


def _execute_backtest(
    prices: pd.DataFrame,
    config: Config,
    per_inst: dict[str, dict[str, pd.Series]],
    annual_vol: dict[str, pd.Series],
    weights: dict[str, float] | pd.DataFrame,
    idm: float | pd.Series,
    fdm: float | pd.Series,
    carry: pd.DataFrame | None,
    regime: pd.Series | None = None,
) -> BacktestResult:
    """Run the rest of the pipeline given pre-computed weights/IDM/FDM.

    `weights`, `idm`, and `fdm` may be static scalars/dicts or daily
    Series/DataFrames produced by expanding-window calibration.
    """
    symbols = list(prices.columns)
    returns = daily_returns(prices)
    index = prices.index

    # Normalise dynamic inputs.
    if isinstance(weights, dict):
        weights_df = pd.DataFrame({s: weights.get(s, 0.0) for s in symbols}, index=index)
    else:
        weights_df = weights.reindex(index=index, columns=symbols).fillna(0.0)
    if isinstance(idm, (int, float)):
        idm_series = pd.Series(float(idm), index=index)
    else:
        idm_series = idm.reindex(index).fillna(1.0)
    if isinstance(fdm, (int, float)):
        fdm_series = pd.Series(float(fdm), index=index)
    else:
        fdm_series = fdm.reindex(index).fillna(1.0)

    # Combined forecast per instrument.
    rule_weights = _rule_weights(sorted({r for d in per_inst.values() for r in d}), config)
    forecasts = pd.DataFrame(
        {
            sym: combine_instrument(rf, rule_weights, fdm_series, config.forecast_cap)
            for sym, rf in per_inst.items()
        }
    )

    # Raw vol-target sizing.
    mult = pd.Series({s: _multiplier(s, config.use_expanded_universe) for s in symbols})
    raw_units = pd.DataFrame(
        {
            sym: position_units(
                forecasts[sym],
                prices[sym],
                annual_vol[sym],
                config.capital,
                config.vol_target,
                weights_df[sym],
                idm_series,
                _multiplier(sym, config.use_expanded_universe),
            )
            for sym in symbols
        }
    )

    # Correlation-spike de-risking overlay.
    if config.use_corr_spike:
        overlay = corr_spike_overlay(
            returns,
            span=config.corr_spike_span,
            threshold=config.corr_spike_threshold,
            max_degross=config.corr_spike_max_degross,
        )
        raw_units = raw_units.mul(overlay, axis=0)
    else:
        overlay = pd.Series(1.0, index=index)

    # Macro regime overlay.
    regime = regime if regime is not None else pd.Series(1.0, index=index)
    if config.use_regime_overlay:
        raw_units = raw_units.mul(regime, axis=0)

    # Realised-vol governor (two-pass; multiplier is lagged → no lookahead).
    cost_bps = _cost_per_symbol(symbols, config)
    if config.use_governor:
        sim_raw = _simulate(
            raw_units, prices, mult, cost_bps, config.capital, config.buffer_fraction
        )
        governor = vol_governor(
            sim_raw["daily"],
            config.vol_target,
            config.governor_span,
            config.governor_min,
            config.governor_max,
            smooth=config.governor_smooth,
        )
        governed = raw_units.mul(governor, axis=0)
    else:
        governor = pd.Series(1.0, index=index)
        governed = raw_units

    sim = _simulate(governed, prices, mult, cost_bps, config.capital, config.buffer_fraction)
    equity = (1.0 + sim["daily"]).cumprod()
    # Recompute the buffered targets for live use (last-row = next-day target).
    buffered = pd.DataFrame(
        {c: apply_buffer(governed[c], config.buffer_fraction) for c in governed.columns},
        index=index,
    )

    # Store the *final* parameter values for reporting/walk-forward reuse.
    final_weights = weights_df.iloc[-1].to_dict() if len(weights_df) else {s: 0.0 for s in symbols}
    final_idm = float(idm_series.iloc[-1]) if len(idm_series) else 1.0
    final_fdm = float(fdm_series.iloc[-1]) if len(fdm_series) else 1.0

    return BacktestResult(
        daily_returns=sim["daily"],
        gross_returns=sim["gross"],
        equity=equity,
        per_instrument_returns=sim["per_inst"],
        per_instrument_gross=sim["per_inst_gross"],
        positions=sim["eff"],
        buffered=buffered,
        notional=sim["notional"],
        forecasts=forecasts,
        turnover=sim["turnover"],
        instrument_corr=returns.corr(),
        governor=governor,
        overlay=overlay,
        regime=regime,
        weights=final_weights,
        idm=final_idm,
        fdm=final_fdm,
        config=config,
    )


def _expanding_calibration(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    per_inst: dict[str, dict[str, pd.Series]],
    config: Config,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Estimate weights, IDM, and FDM on an expanding window with yearly rebal.

    Parameters are estimated using only data available up to each rebal point and
    then applied forward until the next rebal.  This removes the full-sample
    calibration leak from the default backtest path.
    """
    symbols = list(prices.columns)
    dates = prices.index
    n = len(dates)

    min_obs = max(
        config.calibration_min_obs,
        max(fast_slow[1] for fast_slow in config.ewmac_speeds),
        max(config.breakout_spans) if config.use_breakout else 0,
    )
    rebal = max(1, config.calibration_rebal)

    equal_w = {s: 1.0 / len(symbols) for s in symbols} if symbols else {}
    weights_df = pd.DataFrame({s: np.nan for s in symbols}, index=dates)
    idm_series = pd.Series(np.nan, index=dates)
    fdm_series = pd.Series(np.nan, index=dates)

    if n <= min_obs:
        # Not enough history to estimate → neutral parameters.
        return (
            pd.DataFrame(equal_w, index=dates),
            pd.Series(1.0, index=dates),
            pd.Series(1.0, index=dates),
        )

    all_rules = sorted({r for d in per_inst.values() for r in d})
    rule_weights = _rule_weights(all_rules, config)

    weight_records: dict[pd.Timestamp, dict[str, float]] = {}
    idm_records: dict[pd.Timestamp, float] = {}
    fdm_records: dict[pd.Timestamp, float] = {}

    for idx in range(min_obs, n, rebal):
        t = dates[idx]
        past_returns = returns.iloc[:idx]
        past_per_inst = {
            sym: {name: s.iloc[:idx] for name, s in rf.items()} for sym, rf in per_inst.items()
        }
        w = build_instrument_weights(symbols, past_returns, config)
        weight_records[t] = w
        idm_records[t] = estimate_idm(past_returns, w, config.idm_cap)
        rule_corr = pooled_rule_correlation(past_per_inst)
        fdm_records[t] = estimate_fdm(rule_corr, rule_weights, config.fdm_cap)

    if weight_records:
        weights_df.update(pd.DataFrame(weight_records).T)
        weights_df = weights_df.ffill().fillna(pd.Series(equal_w))
    if idm_records:
        idm_series.update(pd.Series(idm_records))
        idm_series = idm_series.ffill().fillna(1.0)
    if fdm_records:
        fdm_series.update(pd.Series(fdm_records))
        fdm_series = fdm_series.ffill().fillna(1.0)

    return weights_df, idm_series, fdm_series


def run_backtest(
    prices: pd.DataFrame,
    config: Config | None = None,
    carry: pd.DataFrame | None = None,
    regime: pd.Series | None = None,
    cot: pd.DataFrame | None = None,
) -> BacktestResult:
    """Run the full backtest pipeline with expanding-window calibration."""
    config = config or Config()
    prices = prices.sort_index().dropna(how="all").copy()
    returns = daily_returns(prices)

    per_inst, annual_vol = _build_forecasts(prices, returns, config, carry, cot)

    # OOS parameter calibration: weights, IDM, and FDM are re-estimated on an
    # expanding window and applied forward only.
    weights_df, idm_series, fdm_series = _expanding_calibration(prices, returns, per_inst, config)

    return _execute_backtest(
        prices, config, per_inst, annual_vol, weights_df, idm_series, fdm_series, carry, regime
    )


def run_backtest_with_params(
    prices: pd.DataFrame,
    config: Config,
    weights: dict[str, float],
    idm: float,
    fdm: float,
    carry: pd.DataFrame | None = None,
    regime: pd.Series | None = None,
    cot: pd.DataFrame | None = None,
) -> BacktestResult:
    """Run the backtest with pre-computed weights, IDM, and FDM.

    Used by walk-forward/purged CV so the OOS fold is evaluated with
    parameters estimated only on the preceding training window.
    """
    prices = prices.sort_index().dropna(how="all").copy()
    returns = daily_returns(prices)
    per_inst, annual_vol = _build_forecasts(prices, returns, config, carry, cot)
    return _execute_backtest(prices, config, per_inst, annual_vol, weights, idm, fdm, carry, regime)
