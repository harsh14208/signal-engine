"""Batch-evaluate a menu of plausible additions to the core strategy.

Run from repo root:
    source .venv/bin/activate && python scripts/eval_options.py

Tests four categories of "things to add":
  1. Orthogonal signals/rules on the core 19 (COT, carry, accel, etc.)
  2. Macro overlays on the core 19 (regime, VIX term, credit, HMM, corr-spike)
  3. Weighting schemes on the core 19
  4. Instrument additions (expanded universe, diversifier ETFs, sectors, etc.)

The honest ranking is by **walk-forward OOS Sharpe** (4-fold purged CV), not
full-sample Sharpe. Reports are saved to data/options_evaluation.json.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_engine.backtest import run_backtest, run_backtest_with_params
from signal_engine.carry_data import build_carry_panel
from signal_engine.config import Config
from signal_engine.cot_data import build_cot_forecast_panel
from signal_engine.data import load_prices
from signal_engine.experiments import log_experiment
from signal_engine.macro import (
    _load_yf_series,
    credit_overlay,
    hmm_regime_overlay,
    load_credit_spread,
    load_vix,
    load_vix_term_structure,
    regime_overlay,
    vix_term_overlay,
)
from signal_engine.markets import symbols as core_symbols
from signal_engine.metrics import sharpe, summary


CORE = core_symbols(expanded=False)
END = "2026-07-13"
CACHE_TAG = "options_experiment"
N_WF_SPLITS = 5  # purged_walk_forward(n_splits=5) yields 4 folds

# Candidate instrument packs (all available on yfinance with reasonable history).
PACK_DIVERSIFIERS = ["BNDX", "PFF", "AMLP", "MUB", "EMLC"]
PACK_RATE = ["BNDX", "MUB"]  # BIL/SHV excluded: cash-like vol breaks vol-targeting
PACK_SEMIS = ["SMH", "SOXX", "XSD"]
PACK_QQQ = ["QQQ"]
PACK_CORE_COMM = ["UNG", "CORN", "WEAT"]


def _all_candidate_symbols() -> list[str]:
    seen = set(CORE)
    for pack in [PACK_DIVERSIFIERS, PACK_RATE, PACK_SEMIS, PACK_QQQ, PACK_CORE_COMM]:
        seen.update(pack)
    # Expanded universe symbols
    from signal_engine.markets import symbols as market_symbols
    seen.update(market_symbols(expanded=True))
    return list(seen)


def _build_carry(prices: pd.DataFrame, cfg: Config) -> pd.DataFrame | None:
    if not (cfg.use_carry or cfg.use_carry_proxies):
        return None
    try:
        return build_carry_panel(prices, cfg)
    except Exception as exc:
        print(f"    ⚠ carry build failed: {exc}")
        return None


def _build_cot(prices: pd.DataFrame, cfg: Config) -> pd.DataFrame | None:
    if not cfg.use_cot:
        return None
    try:
        # Use the expanded COT cache; it covers core + core_commodities + QQQ.
        return build_cot_forecast_panel(prices, expanded=True, tag="expanded", momentum=cfg.cot_momentum)
    except Exception as exc:
        print(f"    ⚠ COT build failed: {exc}")
        return pd.DataFrame(index=prices.index)


def _build_regime(prices: pd.DataFrame, cfg: Config) -> pd.Series | None:
    if not any([
        cfg.use_regime_overlay,
        cfg.use_vix_term_overlay,
        cfg.use_credit_overlay,
        cfg.use_hmm_regime_overlay,
    ]):
        return None
    start = prices.index.min().strftime("%Y-%m-%d")
    end = prices.index.max().strftime("%Y-%m-%d")
    regime = pd.Series(1.0, index=prices.index)
    try:
        if cfg.use_regime_overlay:
            vix = load_vix(start, end)
            mult = regime_overlay(
                prices,
                vix,
                vix_threshold=cfg.regime_threshold,
                max_degear=cfg.regime_max_degear,
            )
            if cfg.regime_smooth is not None and cfg.regime_smooth > 1:
                mult = mult.ewm(span=cfg.regime_smooth, min_periods=1).mean()
            regime *= mult
        if cfg.use_vix_term_overlay:
            vix_df = load_vix_term_structure(start, end)
            mult = vix_term_overlay(
                vix_df,
                short_thresh=cfg.vix_term_short_thresh,
                long_thresh=cfg.vix_term_long_thresh,
                max_gear=cfg.vix_term_max_gear,
                max_degear=cfg.vix_term_max_degear,
            )
            if cfg.vix_term_smooth is not None and cfg.vix_term_smooth > 1:
                mult = mult.ewm(span=cfg.vix_term_smooth, min_periods=1).mean()
            regime *= mult
        if cfg.use_credit_overlay:
            spread = load_credit_spread(start, end).reindex(prices.index).ffill()
            mult = credit_overlay(
                spread,
                upper_thresh=cfg.credit_upper_thresh,
                lower_thresh=cfg.credit_lower_thresh,
                lookback=cfg.credit_lookback,
                max_gear=cfg.credit_max_gear,
                max_degear=cfg.credit_max_degear,
            )
            if cfg.credit_smooth is not None and cfg.credit_smooth > 1:
                mult = mult.ewm(span=cfg.credit_smooth, min_periods=1).mean()
            regime *= mult
        if cfg.use_hmm_regime_overlay:
            hmm_vix = load_vix(start, end)
            spy = _load_yf_series("SPY", start, end)
            tnx = _load_yf_series("^TNX", start, end)
            irx = _load_yf_series("^IRX", start, end)
            mult = hmm_regime_overlay(
                prices,
                hmm_vix,
                spy,
                tnx=tnx,
                irx=irx,
                train_window=cfg.hmm_train_window,
                refit_stride=cfg.hmm_refit_stride,
                bull_thresh=cfg.hmm_bull_thresh,
                bear_thresh=cfg.hmm_bear_thresh,
                trans_thresh=cfg.hmm_trans_thresh,
                bull_gear=cfg.hmm_bull_gear,
                bear_degear=cfg.hmm_bear_degear,
                trans_degear=cfg.hmm_trans_degear,
                random_state=cfg.hmm_random_state,
            )
            if cfg.hmm_smooth is not None and cfg.hmm_smooth > 1:
                mult = mult.ewm(span=cfg.hmm_smooth, min_periods=1).mean()
            regime *= mult
        return regime
    except Exception as exc:
        print(f"    ⚠ regime build failed: {exc}")
        return None


def _purged_walk_forward_ex(
    prices: pd.DataFrame,
    cfg: Config,
    carry: pd.DataFrame | None,
    regime: pd.Series | None,
    cot: pd.DataFrame | None,
    n_splits: int = N_WF_SPLITS,
    embargo_frac: float = 0.02,
) -> dict:
    """Walk-forward that properly slices carry/regime/cot per fold."""
    from signal_engine.metrics import sharpe as _sharpe

    prices = prices.sort_index().dropna(how="all")
    n = len(prices)
    if n_splits < 2 or n < 300:
        return {"insufficient": True, "n": n}

    boundaries = np.linspace(0, n, n_splits + 1, dtype=int)
    embargo = max(1, int(round(n * embargo_frac)))
    folds = []
    min_train = 128

    for i in range(1, n_splits):
        train_end = boundaries[i] - embargo
        test_start = boundaries[i]
        test_end = boundaries[i + 1]
        if train_end < min_train or test_end - test_start < 30:
            continue

        train = prices.iloc[:train_end]
        test = prices.iloc[test_start:test_end]

        train_result = run_backtest(
            train,
            cfg,
            carry=carry.iloc[:train_end] if carry is not None else None,
            regime=regime.iloc[:train_end] if regime is not None else None,
            cot=cot.iloc[:train_end] if cot is not None else None,
        )
        test_result = run_backtest_with_params(
            test,
            cfg,
            train_result.weights,
            train_result.idm,
            train_result.fdm,
            carry=carry.iloc[test_start:test_end] if carry is not None else None,
            regime=regime.iloc[test_start:test_end] if regime is not None else None,
            cot=cot.iloc[test_start:test_end] if cot is not None else None,
        )
        folds.append(
            {
                "train_end": str(train.index[-1]),
                "test_start": str(test.index[0]),
                "test_end": str(test.index[-1]),
                "is_sharpe": _sharpe(train_result.daily_returns),
                "oos_sharpe": _sharpe(test_result.daily_returns),
            }
        )

    if not folds:
        return {"insufficient": True, "n": n}

    gaps = [f["is_sharpe"] - f["oos_sharpe"] for f in folds]
    return {
        "n_folds": len(folds),
        "folds": folds,
        "mean_is_sharpe": float(np.mean([f["is_sharpe"] for f in folds])),
        "mean_oos_sharpe": float(np.mean([f["oos_sharpe"] for f in folds])),
        "mean_gap": float(np.mean(gaps)),
    }


def _avg_offdiag_corr(df: pd.DataFrame) -> float:
    c = df.corr()
    n = c.shape[0]
    if n < 2:
        return float("nan")
    return float(np.nanmean(c.values[np.triu_indices(n, 1)]))


def _run_variant(
    label: str,
    syms: list[str],
    prices: pd.DataFrame,
    cfg: Config,
) -> dict | None:
    panel = prices[syms].dropna(how="all").copy()
    if panel.shape[1] < 2:
        print(f"⚠ {label}: not enough instruments")
        return None

    print(f"\n=== {label} ({len(syms)} instruments) ===")
    carry = _build_carry(panel, cfg)
    regime = _build_regime(panel, cfg)
    cot = _build_cot(panel, cfg)

    try:
        result = run_backtest(panel, cfg, carry=carry, regime=regime, cot=cot)
    except Exception as exc:
        print(f"⚠ {label} backtest failed: {exc}")
        return None

    log_experiment(cfg, result)

    s = summary(result.equity, result.daily_returns, result.turnover)
    print(
        f"  full-sample net SR {s['sharpe']:.2f} | "
        f"IDM {result.idm:.2f} | "
        f"mean corr {_avg_offdiag_corr(result.per_instrument_returns):.2f} | "
        f"days {s['n_days']}"
    )

    wf = _purged_walk_forward_ex(panel, cfg, carry, regime, cot)
    if wf.get("insufficient"):
        print("  walk-forward: insufficient data")
    else:
        print(
            f"  walk-forward IS {wf['mean_is_sharpe']:.2f} | "
            f"OOS {wf['mean_oos_sharpe']:.2f} | gap {wf['mean_gap']:+.2f}"
        )

    return {
        "label": label,
        "symbols": syms,
        "config": cfg.describe(),
        "n_instruments": panel.shape[1],
        "n_days": s["n_days"],
        "start": str(panel.index[0].date()),
        "end": str(panel.index[-1].date()),
        "net_sharpe": s["sharpe"],
        "gross_sharpe": sharpe(result.gross_returns),
        "ann_return": s["ann_return"],
        "ann_vol": s["ann_vol"],
        "max_dd": s["max_drawdown"],
        "calmar": s["calmar"],
        "idm": result.idm,
        "fdm": result.fdm,
        "mean_pairwise_corr": _avg_offdiag_corr(result.per_instrument_returns),
        "diversification_ratio": (
            s["sharpe"] / float(np.nanmean([sharpe(result.per_instrument_returns[c]) for c in result.per_instrument_returns.columns]))
            if float(np.nanmean([sharpe(result.per_instrument_returns[c]) for c in result.per_instrument_returns.columns])) not in (0, np.nan)
            else float("nan")
        ),
        "wf_mean_is": wf.get("mean_is_sharpe"),
        "wf_mean_oos": wf.get("mean_oos_sharpe"),
        "wf_gap": wf.get("mean_gap"),
        "wf_folds": wf.get("folds"),
    }


def main() -> int:
    all_syms = _all_candidate_symbols()
    print(f"Fetching/loading prices for {len(all_syms)} symbols...")
    prices = load_prices(
        all_syms,
        start="2007-01-01",
        end=END,
        source="yfinance",
        cache_tag=CACHE_TAG,
    )
    print(f"Loaded {prices.shape[1]} symbols, {len(prices)} rows, {prices.index[0].date()} to {prices.index[-1].date()}")

    from signal_engine.markets import symbols as market_symbols
    expanded_syms = market_symbols(expanded=True)

    variants = [
        # Baseline
        ("core_19_baseline", CORE, Config()),

        # Signals / rules
        ("+ COT", CORE, Config(use_cot=True)),
        ("+ carry proxies", CORE, Config(use_carry_proxies=True)),
        ("+ real bond carry", CORE, Config(use_carry_proxies=True, use_real_bond_carry=True)),
        ("+ acceleration", CORE, Config(use_accel=True)),
        ("+ cross-sectional momentum", CORE, Config(use_xsmom=True)),
        ("+ network momentum", CORE, Config(use_network_momentum=True)),
        ("+ COT + carry", CORE, Config(use_cot=True, use_carry_proxies=True)),
        ("+ COT + acceleration", CORE, Config(use_cot=True, use_accel=True)),

        # Overlays
        ("+ regime overlay", CORE, Config(use_regime_overlay=True)),
        ("+ VIX term overlay", CORE, Config(use_vix_term_overlay=True)),
        ("+ credit overlay", CORE, Config(use_credit_overlay=True)),
        ("+ HMM regime overlay", CORE, Config(use_hmm_regime_overlay=True)),
        ("+ correlation-spike overlay", CORE, Config(use_corr_spike=True)),

        # Weighting schemes
        ("weight: cluster", CORE, Config(weight_scheme="cluster")),
        ("weight: corr-cluster", CORE, Config(weight_scheme="corr_cluster")),
        ("weight: sharpe", CORE, Config(weight_scheme="sharpe")),

        # Instrument additions
        ("expanded universe", expanded_syms, Config(use_expanded_universe=True)),
        ("+ core commodities", CORE + PACK_CORE_COMM, Config()),
        ("+ QQQ", CORE + PACK_QQQ, Config()),
        ("+ semis (SMH/SOXX/XSD)", CORE + PACK_SEMIS, Config()),
        ("+ diversifier pack (BNDX/PFF/AMLP/MUB/EMLC)", CORE + PACK_DIVERSIFIERS, Config()),
        ("+ rate pack (BNDX/MUB)", CORE + PACK_RATE, Config()),

        # Combinations that look promising
        ("+ diversifier pack + COT", CORE + PACK_DIVERSIFIERS, Config(use_cot=True)),
        ("+ diversifier pack + carry", CORE + PACK_DIVERSIFIERS, Config(use_carry_proxies=True)),
        ("+ COT + corr-cluster weights", CORE, Config(use_cot=True, weight_scheme="corr_cluster")),
        ("+ COT + carry + diversifier pack", CORE + PACK_DIVERSIFIERS, Config(use_cot=True, use_carry_proxies=True)),
    ]

    results = []
    for label, syms, cfg in variants:
        try:
            res = _run_variant(label, syms, prices, cfg)
            if res:
                results.append(res)
        except Exception as exc:
            print(f"⚠ {label} failed: {exc}")

    # Rank by walk-forward OOS; fall back to full-sample if WF missing.
    def _sort_key(r: dict) -> float:
        oos = r.get("wf_mean_oos")
        if oos is not None and not np.isnan(oos):
            return oos
        return r.get("net_sharpe", float("-inf"))

    results.sort(key=_sort_key, reverse=True)

    print("\n\n## Ranking by walk-forward OOS Sharpe")
    df = pd.DataFrame(
        [
            {
                "rank": i + 1,
                "variant": r["label"],
                "n_instr": r["n_instruments"],
                "days": r["n_days"],
                "net_SR": r["net_sharpe"],
                "gross_SR": r["gross_sharpe"],
                "max_DD": r["max_dd"],
                "Calmar": r["calmar"],
                "IDM": r["idm"],
                "mean_corr": r["mean_pairwise_corr"],
                "WF_IS": r["wf_mean_is"],
                "WF_OOS": r["wf_mean_oos"],
                "WF_gap": r["wf_gap"],
            }
            for i, r in enumerate(results)
        ]
    )
    # Formatting
    fmt = df.copy()
    for c in ["net_SR", "gross_SR", "max_DD", "Calmar", "IDM", "mean_corr", "WF_IS", "WF_OOS", "WF_gap"]:
        fmt[c] = fmt[c].map(lambda x: f"{x:.2f}" if isinstance(x, float) and not np.isnan(x) else "—")
    print(fmt.to_string(index=False))

    report_path = os.path.join("data", "options_evaluation.json")
    with open(report_path, "w") as f:
        json.dump(
            {
                "run_at": datetime.now(timezone.utc).isoformat(),
                "n_symbols_loaded": prices.shape[1],
                "start": str(prices.index[0].date()),
                "end": str(prices.index[-1].date()),
                "ranked_results": results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nSaved report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
