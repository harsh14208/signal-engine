"""Command line: load data → backtest → report → (optional) validate.

python -m signal_engine                      # synthetic demo (offline)
python -m signal_engine --validate           # + Lo CI, Deflated SR, MC, placebo
python -m signal_engine --source yfinance     # real ETF-proxy data (needs network)
python -m signal_engine --oos 0.7 --validate  # chronological IS/OOS split
python -m signal_engine --walk-forward 5      # purged expanding-window CV
python -m signal_engine --diagnostics         # cost/buffer frontier + attribution
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .carry_data import build_carry_panel
from .cot_data import build_cot_forecast_panel
from .curve_data import load_curve_instruments
from .equity_momentum_sleeve import build_equity_momentum_sleeve
from .config import Config, DEFAULT_BREAKOUT_SPANS
from .data import load_prices, synthetic_carry
from .diagnostics import cost_buffer_frontier, per_instrument_attribution, vix_regime_split
from .experiments import count_experiments, log_experiment
from .macro import (
    credit_overlay,
    hmm_regime_overlay,
    load_credit_spread,
    load_vix,
    load_vix_term_structure,
    regime_overlay,
    vix_term_overlay,
)
from .markets import symbols
from .metrics import sharpe
from .monitor import edge_decay_report
from .report import full_report
from .validation import block_bootstrap_sharpe, lo_sharpe_ci, placebo_sharpes, purged_walk_forward


_FLAG_TAXONOMY = """\
flag taxonomy (validated 2026-07-15 — promote on the walk-forward, never one split):
  CORE (shape the validated default): --vol-target --buffer --cost-scheme
      --weight-scheme --no-governor --governor-smooth
  VALIDATED-POSITIVE (opt-in; clears the walk-forward, safe to ship):
      --cot   (CFTC positioning; full 0.69→0.72, walk-forward OOS 0.61→0.63)
      --network-momentum  (lead-lag price graph signal; full 0.69→0.73, WF OOS 0.63→0.67)
      --semis  (add SMH/SOXX/XSD; WF OOS 0.63→0.69, no leverage blow-up)
      --qqq    (add Nasdaq-100; WF OOS 0.63→0.68, no leverage blow-up)
  RESEARCH — tested, NONE beat the walk-forward default; opt-in only:
      --cluster-weights --expanded-universe --empirical-scalars --regime-overlay
      --vix-term-overlay --credit-overlay --hmm-regime-overlay
      --equity-momentum-sleeve --curve-steepener --real-bond-carry --garch-vol
      --accel --xsmom --corr-spike --carry-proxies --core-commodities --cot-momentum
      --ship-candidate
  LEVERAGE-DEPENDENT — require --financing-rate for honest comparison:
      --diversifier-pack --rate-pack  (bond packs; edge shrinks under realistic financing)
      --weight-scheme corr_cluster    (concentrates in low-vol instruments; OOS 0.54→0.41 at 1% financing)
  NEW OPT-IN OVERLAYS (validate before promoting):
      --drawdown-control              (de-risk on deep drawdowns; easy to overfit threshold)
      --trend-strength-filter         (de-gear when combined forecasts are historically weak)
      --calibration-smooth N          (blend weights/IDM/FDM transitions to cut turnover)
      --alarm-on-worst-quartile       (self-calibrating edge-decay kill switch)
  VALIDATION / DIAGNOSTICS: --validate --oos --walk-forward --diagnostics
      --monitor --n-trials --placebo
"""


def build_config(args) -> Config:
    return Config(
        capital=args.capital,
        vol_target=args.vol_target,
        use_breakout=not args.no_breakout,
        breakout_spans=args.breakout_spans if args.breakout_spans is not None else DEFAULT_BREAKOUT_SPANS,
        use_carry=args.carry,
        use_carry_proxies=args.carry_proxies,
        use_real_bond_carry=args.real_bond_carry,
        use_curve_steepener=args.curve_steepener,
        curve_steepener_scale=args.curve_steepener_scale,
        curve_steepener_cost_bps=args.curve_steepener_cost_bps,
        use_equity_momentum_sleeve=args.equity_momentum_sleeve,
        eq_mom_lookback=args.eq_mom_lookback,
        eq_mom_rebalance=args.eq_mom_rebalance,
        eq_mom_decile=args.eq_mom_decile,
        use_expanded_universe=args.expanded_universe,
        use_empirical_scalars=args.empirical_scalars,
        use_regime_overlay=args.regime_overlay,
        regime_threshold=args.regime_threshold,
        regime_max_degear=args.regime_max_degear,
        use_vix_term_overlay=args.vix_term_overlay,
        vix_term_short_thresh=args.vix_term_short_thresh,
        vix_term_long_thresh=args.vix_term_long_thresh,
        vix_term_max_gear=args.vix_term_max_gear,
        vix_term_max_degear=args.vix_term_max_degear,
        vix_term_smooth=args.vix_term_smooth,
        use_credit_overlay=args.credit_overlay,
        credit_upper_thresh=args.credit_upper_thresh,
        credit_lower_thresh=args.credit_lower_thresh,
        credit_lookback=args.credit_lookback,
        credit_max_gear=args.credit_max_gear,
        credit_max_degear=args.credit_max_degear,
        credit_smooth=args.credit_smooth,
        cost_bps=args.cost_bps,
        cost_scheme=args.cost_scheme,
        buffer_fraction=args.buffer,
        calibration_smooth=args.calibration_smooth,
        weight_scheme=args.weight_scheme,
        cluster_weights=args.cluster_weights,
        use_governor=not args.no_governor,
        governor_smooth=args.governor_smooth,
        regime_smooth=args.regime_smooth,
        use_accel=args.accel,
        use_xsmom=args.xsmom,
        use_corr_spike=args.corr_spike,
        corr_spike_span=args.corr_spike_span,
        corr_spike_threshold=args.corr_spike_threshold,
        corr_spike_max_degross=args.corr_spike_max_degross,
        use_cot=args.cot,
        cot_momentum=args.cot_momentum,
        use_core_commodities=args.core_commodities,
        max_gross_notional=args.max_gross,
        financing_rate=args.financing_rate,
        financing_threshold=args.financing_threshold,
        max_annual_financing_cost=args.max_annual_financing_cost,
        use_drawdown_control=args.use_drawdown_control,
        drawdown_threshold=args.drawdown_threshold,
        drawdown_scale=args.drawdown_scale,
        drawdown_recovery=args.drawdown_recovery,
        use_trend_strength_filter=args.use_trend_strength_filter,
        trend_strength_window=args.trend_strength_window,
        trend_strength_threshold=args.trend_strength_threshold,
        trend_strength_scale=args.trend_strength_scale,
        use_crypto=args.use_crypto,
        crypto_max_risk_weight=(
            args.crypto_max_risk_weight if args.crypto_max_risk_weight is not None else 0.05
        ),
        use_curated_breadth=args.use_curated_breadth,
        use_fx_carry=args.use_fx_carry,
        use_garch_vol=args.use_garch_vol,
        garch_weight=args.garch_weight,
        garch_min_history=args.garch_min_history,
        garch_refit_step=args.garch_refit_step,
        garch_horizon=args.garch_horizon,
        use_hmm_regime_overlay=args.use_hmm_regime_overlay,
        hmm_train_window=args.hmm_train_window,
        hmm_refit_stride=args.hmm_refit_stride,
        hmm_bull_thresh=args.hmm_bull_thresh,
        hmm_bear_thresh=args.hmm_bear_thresh,
        hmm_trans_thresh=args.hmm_trans_thresh,
        hmm_bull_gear=args.hmm_bull_gear,
        hmm_bear_degear=args.hmm_bear_degear,
        hmm_trans_degear=args.hmm_trans_degear,
        hmm_smooth=args.hmm_smooth,
        hmm_random_state=args.hmm_random_state,
        use_network_momentum=args.use_network_momentum,
        nm_speed=args.nm_speed,
        nm_lookback=args.nm_lookback,
        nm_lag=args.nm_lag,
        nm_rebal=args.nm_rebal,
        nm_top_k=args.nm_top_k,
    )


def _n_trials_for(args) -> int:
    """Use the real experiment count unless the user overrode it."""
    if args.n_trials is not None:
        return args.n_trials
    return max(2, count_experiments() + 1)


def _print_validation(result, cfg, args) -> None:
    print("\n## Statistical validation\n")
    n_trials = _n_trials_for(args)
    lo = lo_sharpe_ci(result.daily_returns, n_trials=n_trials)
    if lo.get("insufficient"):
        print("- Insufficient data for Lo CI.")
    else:
        verdict = "✅ outside" if not lo["zero_inside"] else "⚠ INSIDE"
        print(
            f"- **Lo (2002) 95% CI (annualised):** [{lo['ci_low']:.2f}, {lo['ci_high']:.2f}] "
            f"— SR=0 is {verdict} (N={lo['n']} days)"
        )
        dverdict = "✅ clears" if lo["passes_deflated"] else "⚠ FAILS"
        print(
            f"- **Deflated Sharpe ({lo['n_trials']} trials):** expected max by chance = "
            f"{lo['deflated_expected_max']:.2f} → Sharpe {lo['sharpe']:.2f} {dverdict} it"
        )

    mc = block_bootstrap_sharpe(result.daily_returns)
    if not mc.get("insufficient"):
        edge = "✅ > 0" if mc["edge_real"] else "⚠ ≤ 0"
        print(
            f"- **Block-bootstrap MC (block={mc['block']}):** P5={mc['p5']:.2f} / "
            f"P50={mc['p50']:.2f} / P95={mc['p95']:.2f} — P5 {edge}"
        )

    # Random-walk placebo: same strategy on driftless panels = noise floor.
    n_days = min(len(result.daily_returns), 2500)
    pl = placebo_sharpes(
        lambda panel: run_backtest(panel, cfg).daily_returns,
        n_placebo=args.placebo,
        n_instruments=result.per_instrument_returns.shape[1],
        n_days=n_days,
    )
    port = sharpe(result.daily_returns)
    clears = "✅ clears" if port > pl["noise_floor_95"] else "⚠ does NOT clear"
    print(
        f"- **Random-walk placebo (n={pl['n_placebo']}):** noise floor "
        f"mean={pl['mean']:.2f}, 95th pct={pl['noise_floor_95']:.2f} → "
        f"real Sharpe {port:.2f} {clears} the floor"
    )


def _print_oos(result, frac: float) -> None:
    d = result.daily_returns.dropna()
    split = int(len(d) * frac)
    is_sr, oos_sr = sharpe(d.iloc[:split]), sharpe(d.iloc[split:])
    print("\n## Out-of-sample (chronological split)\n")
    print(f"- IS Sharpe (first {frac:.0%}): **{is_sr:.2f}**")
    print(f"- OOS Sharpe (last {1 - frac:.0%}): **{oos_sr:.2f}**")
    gap = is_sr - oos_sr
    flag = "✅ small" if abs(gap) < 0.2 else "⚠ large"
    print(f"- Curation gap (IS − OOS): **{gap:+.2f}** ({flag})")


def _print_walk_forward(prices, cfg, n_splits: int, carry=None, cot=None) -> None:
    wf = purged_walk_forward(prices, cfg, n_splits=n_splits, carry=carry, cot=cot)
    print("\n## Walk-forward / purged CV\n")
    if wf.get("insufficient"):
        print("- Insufficient data for walk-forward analysis.")
        return
    print(f"- Folds: {wf['n_folds']}  mean IS Sharpe: **{wf['mean_is_sharpe']:.2f}**")
    print(
        f"- mean OOS Sharpe: **{wf['mean_oos_sharpe']:.2f}**  mean gap: **{wf['mean_gap']:+.2f}**"
    )
    for f in wf["folds"]:
        print(
            f"  - {f['test_start']} → {f['test_end']}: "
            f"IS {f['is_sharpe']:.2f} / OOS {f['oos_sharpe']:.2f}"
        )


def _print_diagnostics(prices, result, cfg) -> None:
    print("\n## Diagnostics\n")
    print("### Cost × buffer frontier (70/30 IS/OOS)\n")
    frontier = cost_buffer_frontier(prices, cfg)
    print(frontier.to_string(index=False))

    print("\n### Per-instrument attribution\n")
    attr = per_instrument_attribution(result)
    print(attr.to_string())

    print("\n### VIX regime split\n")
    try:
        vix = load_vix(
            prices.index.min().strftime("%Y-%m-%d"), prices.index.max().strftime("%Y-%m-%d")
        )
        split = vix_regime_split(result.daily_returns, vix)
        print(f"- Median VIX: {split['median_vix']:.1f}")
        for label, stats in [("High VIX", split["high_vix"]), ("Low VIX", split["low_vix"])]:
            print(
                f"- {label}: n={stats['n_days']}, Sharpe={stats['sharpe']:.2f}, "
                f"vol={stats['ann_vol']:.1%}, ret={stats['ann_return']:.1%}, MaxDD={stats['max_dd']:.1%}"
            )
    except Exception as exc:
        print(f"- Could not load VIX for regime split: {exc}")


def _print_monitor(result) -> None:
    rep = edge_decay_report(result.daily_returns)
    print("\n## Edge-decay monitor (rolling 1y Sharpe)\n")
    if rep.get("insufficient"):
        print("- Insufficient data.")
        return
    print(
        f"- current **{rep['current']:.2f}**  median {rep['median']:.2f}  "
        f"min {rep['min']:.2f}  max {rep['max']:.2f}  "
        f"% windows <0: {rep['pct_windows_below_zero']:.0%}"
    )
    flag = "⚠ ALARM (below floor)" if rep["alarm"] else "✅ healthy"
    print(f"- latest vs floor {rep['alarm_floor']:.2f}: {flag}")


def _build_symbol_list(args) -> tuple[list[str], str]:
    """Return the symbol list and cache tag implied by the CLI arguments."""
    expanded = args.expanded_universe
    if args.core_commodities and not expanded:
        syms = symbols(expanded=False) + ["UNG", "CORN", "WEAT"]
        cache_tag = "core_plus"
    else:
        syms = symbols(expanded=expanded)
        cache_tag = "expanded" if expanded else "universe"

    pack_additions: list[str] = []
    if args.semis:
        pack_additions += ["SMH", "SOXX", "XSD"]
    if args.qqq:
        pack_additions += ["QQQ"]
    if args.diversifier_pack:
        pack_additions += ["BNDX", "PFF", "AMLP", "MUB", "EMLC"]
    if args.rate_pack:
        pack_additions += ["BNDX", "MUB"]
    if args.use_crypto:
        pack_additions += ["BTC-USD", "ETH-USD"]
    if args.use_curated_breadth:
        pack_additions += [
            "UNG", "CPER", "CORN", "WEAT", "SOYB",
            "EMB", "BWX", "FXA", "FXB", "FXC",
        ]
    if pack_additions:
        seen = set(syms)
        syms = syms + [s for s in pack_additions if s not in seen]
        cache_tag = "options_experiment"

    return syms, cache_tag


def run(args) -> int:
    cfg = build_config(args)
    syms, cache_tag = _build_symbol_list(args)

    prices = load_prices(
        syms, start=args.start, end=args.end, source=args.source, cache_tag=cache_tag
    )
    if cfg.use_curve_steepener:
        curve_prices = load_curve_instruments(prices, scale=cfg.curve_steepener_scale)
        prices = pd.concat([prices, curve_prices], axis=1)
    if cfg.use_equity_momentum_sleeve:
        eq_mom = build_equity_momentum_sleeve(
            prices.index.min().strftime("%Y-%m-%d"),
            prices.index.max().strftime("%Y-%m-%d"),
            lookback=cfg.eq_mom_lookback,
            rebalance=cfg.eq_mom_rebalance,
            decile=cfg.eq_mom_decile,
        )
        prices = pd.concat([prices, eq_mom.to_frame()], axis=1)
    if prices.shape[1] < 2:
        print("Not enough instruments with data.")
        return 1

    carry = None
    if args.carry:
        # Demo carry only. Real carry needs term-structure data — see README §carry.
        carry = synthetic_carry(list(prices.columns), prices.index)
    elif cfg.use_carry_proxies:
        carry = build_carry_panel(prices, cfg)

    regime = None
    if cfg.use_regime_overlay:
        vix = load_vix(
            prices.index.min().strftime("%Y-%m-%d"), prices.index.max().strftime("%Y-%m-%d")
        )
        regime = regime_overlay(
            prices,
            vix,
            vix_threshold=cfg.regime_threshold,
            max_degear=cfg.regime_max_degear,
        )
        if cfg.regime_smooth is not None and cfg.regime_smooth > 1:
            regime = regime.ewm(span=cfg.regime_smooth, min_periods=1).mean()

    if cfg.use_vix_term_overlay:
        vix_df = load_vix_term_structure(
            prices.index.min().strftime("%Y-%m-%d"), prices.index.max().strftime("%Y-%m-%d")
        )
        vix_mult = vix_term_overlay(
            vix_df,
            short_thresh=cfg.vix_term_short_thresh,
            long_thresh=cfg.vix_term_long_thresh,
            max_gear=cfg.vix_term_max_gear,
            max_degear=cfg.vix_term_max_degear,
        )
        if cfg.vix_term_smooth is not None and cfg.vix_term_smooth > 1:
            vix_mult = vix_mult.ewm(span=cfg.vix_term_smooth, min_periods=1).mean()
        if regime is None:
            regime = vix_mult
        else:
            regime = regime * vix_mult

    if cfg.use_credit_overlay:
        spread = (
            load_credit_spread(
                prices.index.min().strftime("%Y-%m-%d"), prices.index.max().strftime("%Y-%m-%d")
            )
            .reindex(prices.index)
            .ffill()
        )
        credit_mult = credit_overlay(
            spread,
            upper_thresh=cfg.credit_upper_thresh,
            lower_thresh=cfg.credit_lower_thresh,
            lookback=cfg.credit_lookback,
            max_gear=cfg.credit_max_gear,
            max_degear=cfg.credit_max_degear,
        )
        if cfg.credit_smooth is not None and cfg.credit_smooth > 1:
            credit_mult = credit_mult.ewm(span=cfg.credit_smooth, min_periods=1).mean()
        if regime is None:
            regime = credit_mult
        else:
            regime = regime * credit_mult

    if cfg.use_hmm_regime_overlay:
        from .macro import _load_yf_series

        hmm_vix = load_vix(
            prices.index.min().strftime("%Y-%m-%d"), prices.index.max().strftime("%Y-%m-%d")
        )
        spy = _load_yf_series(
            "SPY", prices.index.min().strftime("%Y-%m-%d"), prices.index.max().strftime("%Y-%m-%d")
        )
        tnx = _load_yf_series(
            "^TNX", prices.index.min().strftime("%Y-%m-%d"), prices.index.max().strftime("%Y-%m-%d")
        )
        irx = _load_yf_series(
            "^IRX", prices.index.min().strftime("%Y-%m-%d"), prices.index.max().strftime("%Y-%m-%d")
        )
        hmm_mult = hmm_regime_overlay(
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
            hmm_mult = hmm_mult.ewm(span=cfg.hmm_smooth, min_periods=1).mean()
        if regime is None:
            regime = hmm_mult
        else:
            regime = regime * hmm_mult

    cot_tag = {"universe": "core", "core_plus": "core_plus", "expanded": "expanded"}.get(
        cache_tag, "core"
    )
    cot = (
        build_cot_forecast_panel(prices, tag=cot_tag, momentum=cfg.cot_momentum)
        if cfg.use_cot
        else None
    )
    result = run_backtest(prices, cfg, carry=carry, regime=regime, cot=cot)
    print(f"# signal-engine — {args.source} run\n")
    print(full_report(result))
    if args.oos:
        _print_oos(result, args.oos)
    if args.walk_forward:
        _print_walk_forward(prices, cfg, args.walk_forward, carry=carry, cot=cot)
    if args.diagnostics:
        _print_diagnostics(prices, result, cfg)
    if args.monitor:
        _print_monitor(result)
    if args.validate:
        _print_validation(result, cfg, args)

    # Log this experiment so the Deflated-Sharpe counter reflects real search.
    log_experiment(cfg, result)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="signal_engine",
        description=__doc__,
        epilog=_FLAG_TAXONOMY,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--source", default="synthetic", choices=["synthetic", "yfinance", "cache", "auto"]
    )
    p.add_argument("--start", default="2007-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--capital", type=float, default=1_000_000.0)
    p.add_argument("--vol-target", type=float, default=0.20, dest="vol_target")
    p.add_argument("--cost-bps", type=float, default=1.5, dest="cost_bps")
    p.add_argument(
        "--cost-scheme",
        default="flat",
        choices=["flat", "instrument"],
        dest="cost_scheme",
        help="flat 1.5 bps or per-instrument spreads",
    )
    p.add_argument("--buffer", type=float, default=0.30)
    p.add_argument(
        "--calibration-smooth",
        type=int,
        default=None,
        dest="calibration_smooth",
        help="smooth weights/IDM/FDM transitions over N days to cut estimation-driven turnover",
    )
    p.add_argument("--no-breakout", action="store_true")
    p.add_argument(
        "--breakout-spans",
        type=lambda s: tuple(int(x.strip()) for x in s.split(",")),
        default=None,
        dest="breakout_spans",
        help="comma-separated breakout channel spans (default: 40,80,160)",
    )
    p.add_argument(
        "--weight-scheme",
        default="equal",
        choices=["equal", "cluster", "corr_cluster", "sharpe"],
        dest="weight_scheme",
        help="instrument weighting scheme",
    )
    p.add_argument(
        "--cluster-weights",
        action="store_true",
        dest="cluster_weights",
        help="asset-class cluster weights (research; OFF by default — it hurt on this universe)",
    )
    p.add_argument(
        "--no-governor",
        action="store_true",
        dest="no_governor",
        help="disable the realised-vol governor (on by default)",
    )
    p.add_argument(
        "--governor-smooth",
        type=int,
        default=None,
        dest="governor_smooth",
        help="EWMA span for the governor multiplier (None = raw)",
    )
    p.add_argument("--carry", action="store_true", help="demo carry rule (synthetic series)")
    p.add_argument(
        "--carry-proxies",
        action="store_true",
        dest="carry_proxies",
        help="use free bond/equity carry proxies (FRED + yfinance dividends)",
    )
    p.add_argument(
        "--real-bond-carry",
        action="store_true",
        dest="real_bond_carry",
        help="use tenor-specific Treasury roll-down carry (DGS2/5/10/30) instead of T10Y3M slope",
    )
    p.add_argument(
        "--curve-steepener",
        action="store_true",
        dest="curve_steepener",
        help="add a synthetic UST 2s10s curve-steepener instrument",
    )
    p.add_argument(
        "--curve-steepener-scale",
        type=float,
        default=1.0,
        dest="curve_steepener_scale",
        help="scalar on the 2s10s spread-change return",
    )
    p.add_argument(
        "--curve-steepener-cost-bps",
        type=float,
        default=0.5,
        dest="curve_steepener_cost_bps",
        help="per-side cost for the synthetic curve steepener",
    )
    p.add_argument(
        "--equity-momentum-sleeve",
        action="store_true",
        dest="equity_momentum_sleeve",
        help="add a synthetic S&P 500 cross-sectional momentum sleeve",
    )
    p.add_argument("--eq-mom-lookback", type=int, default=252, dest="eq_mom_lookback")
    p.add_argument("--eq-mom-rebalance", type=int, default=21, dest="eq_mom_rebalance")
    p.add_argument(
        "--eq-mom-decile",
        type=float,
        default=0.10,
        dest="eq_mom_decile",
        help="top/bottom decile cut for the momentum sleeve",
    )
    p.add_argument(
        "--expanded-universe",
        action="store_true",
        dest="expanded_universe",
        help="use the ~40-name free ETF universe",
    )
    p.add_argument(
        "--empirical-scalars",
        action="store_true",
        dest="empirical_scalars",
        help="estimate rule scalars from an expanding window of the cached data",
    )
    p.add_argument(
        "--regime-overlay",
        action="store_true",
        dest="regime_overlay",
        help="scale trend exposure down in high VIX / high-drawdown regimes",
    )
    p.add_argument("--regime-threshold", type=float, default=20.0, dest="regime_threshold")
    p.add_argument("--regime-max-degear", type=float, default=0.5, dest="regime_max_degear")
    p.add_argument(
        "--regime-smooth",
        type=int,
        default=None,
        dest="regime_smooth",
        help="EWMA span for the regime de-gross multiplier (None = raw)",
    )
    p.add_argument(
        "--vix-term-overlay",
        action="store_true",
        dest="vix_term_overlay",
        help="use VIX term-structure (9D/spot and 3M/spot ratios) gear/de-gear overlay",
    )
    p.add_argument(
        "--vix-term-short-thresh", type=float, default=1.10, dest="vix_term_short_thresh"
    )
    p.add_argument("--vix-term-long-thresh", type=float, default=0.95, dest="vix_term_long_thresh")
    p.add_argument("--vix-term-max-gear", type=float, default=1.25, dest="vix_term_max_gear")
    p.add_argument("--vix-term-max-degear", type=float, default=0.50, dest="vix_term_max_degear")
    p.add_argument(
        "--vix-term-smooth",
        type=int,
        default=None,
        dest="vix_term_smooth",
        help="EWMA span for the VIX term-structure multiplier (None = raw)",
    )
    p.add_argument(
        "--credit-overlay",
        action="store_true",
        dest="credit_overlay",
        help="use Baa-10Y credit-spread gear/de-gear overlay",
    )
    p.add_argument("--credit-upper-thresh", type=float, default=1.50, dest="credit_upper_thresh")
    p.add_argument("--credit-lower-thresh", type=float, default=0.80, dest="credit_lower_thresh")
    p.add_argument(
        "--credit-lookback",
        type=int,
        default=1260,
        dest="credit_lookback",
        help="rolling lookback (days) for the credit-spread median",
    )
    p.add_argument("--credit-max-gear", type=float, default=1.25, dest="credit_max_gear")
    p.add_argument("--credit-max-degear", type=float, default=0.50, dest="credit_max_degear")
    p.add_argument(
        "--credit-smooth",
        type=int,
        default=None,
        dest="credit_smooth",
        help="EWMA span for the credit multiplier (None = raw)",
    )
    p.add_argument(
        "--hmm-regime-overlay",
        action="store_true",
        dest="use_hmm_regime_overlay",
        help="use a 2-state HMM macro-regime overlay (VIX/SPY/yield-curve/realised-vol)",
    )
    p.add_argument("--hmm-train-window", type=int, default=252, dest="hmm_train_window")
    p.add_argument("--hmm-refit-stride", type=int, default=63, dest="hmm_refit_stride")
    p.add_argument("--hmm-bull-thresh", type=float, default=0.75, dest="hmm_bull_thresh")
    p.add_argument("--hmm-bear-thresh", type=float, default=0.70, dest="hmm_bear_thresh")
    p.add_argument("--hmm-trans-thresh", type=float, default=0.15, dest="hmm_trans_thresh")
    p.add_argument("--hmm-bull-gear", type=float, default=1.10, dest="hmm_bull_gear")
    p.add_argument("--hmm-bear-degear", type=float, default=0.70, dest="hmm_bear_degear")
    p.add_argument("--hmm-trans-degear", type=float, default=0.85, dest="hmm_trans_degear")
    p.add_argument(
        "--hmm-smooth",
        type=int,
        default=None,
        dest="hmm_smooth",
        help="EWMA span for the HMM regime multiplier (None = raw)",
    )
    p.add_argument("--hmm-random-state", type=int, default=42, dest="hmm_random_state")
    p.add_argument("--accel", action="store_true", help="add acceleration rule")
    p.add_argument("--xsmom", action="store_true", help="add cross-sectional momentum rule")
    p.add_argument("--validate", action="store_true", help="run the statistical honesty suite")
    p.add_argument(
        "--oos",
        type=float,
        default=None,
        metavar="FRAC",
        help="chronological IS/OOS split, e.g. 0.7",
    )
    p.add_argument(
        "--walk-forward",
        type=int,
        default=None,
        metavar="N",
        dest="walk_forward",
        help="purged expanding-window CV with N splits",
    )
    p.add_argument(
        "--diagnostics",
        action="store_true",
        help="print cost/buffer frontier, attribution, and VIX regime split",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=None,
        dest="n_trials",
        help="Deflated-Sharpe trial count (default: real experiment log count)",
    )
    p.add_argument(
        "--corr-spike",
        action="store_true",
        dest="corr_spike",
        help="enable correlation-spike de-risking overlay",
    )
    p.add_argument("--corr-spike-span", type=int, default=60, dest="corr_spike_span")
    p.add_argument(
        "--corr-spike-threshold",
        type=float,
        default=0.50,
        dest="corr_spike_threshold",
    )
    p.add_argument(
        "--corr-spike-max-degross",
        type=float,
        default=0.50,
        dest="corr_spike_max_degross",
    )
    p.add_argument(
        "--ship-candidate",
        action="store_true",
        dest="ship_candidate",
        help="RESEARCH preset (expanded universe + regime overlay + 30%% buffer): higher "
        "full-sample Sharpe but NOT better on walk-forward OOS than the default — not promoted (see README)",
    )
    p.add_argument(
        "--cot",
        action="store_true",
        help="COT positioning forecast (free CFTC data; contrarian/commercial-value sign)",
    )
    p.add_argument(
        "--cot-momentum",
        action="store_true",
        dest="cot_momentum",
        help="A/B research: flip the COT sign to momentum (follow specs) vs contrarian",
    )
    p.add_argument(
        "--network-momentum",
        action="store_true",
        dest="use_network_momentum",
        help="lead-lag price-graph momentum signal (opt-in; validated positive)",
    )
    p.add_argument("--nm-speed", type=int, nargs=2, default=(32, 128), dest="nm_speed")
    p.add_argument("--nm-lookback", type=int, default=256, dest="nm_lookback")
    p.add_argument("--nm-lag", type=int, default=1, dest="nm_lag")
    p.add_argument("--nm-rebal", type=int, default=63, dest="nm_rebal")
    p.add_argument("--nm-top-k", type=int, default=3, dest="nm_top_k")
    p.add_argument(
        "--core-commodities",
        action="store_true",
        dest="core_commodities",
        help="core + UNG/CORN/WEAT (the COT-mapped diversifying commodities)",
    )
    p.add_argument(
        "--semis",
        action="store_true",
        help="add SMH/SOXX/XSD semiconductor ETFs to the universe",
    )
    p.add_argument(
        "--qqq",
        action="store_true",
        help="add QQQ (Nasdaq-100) to the universe",
    )
    p.add_argument(
        "--crypto",
        action="store_true",
        dest="use_crypto",
        help="add BTC-USD and ETH-USD to the universe (opt-in; high vol, short history)",
    )
    p.add_argument(
        "--crypto-max-risk-weight",
        type=float,
        default=None,
        dest="crypto_max_risk_weight",
        help="max portfolio risk contribution per crypto instrument (default 0.05)",
    )
    p.add_argument(
        "--curated-breadth",
        action="store_true",
        dest="use_curated_breadth",
        help="add correlation-selected breadth instruments (UNG/CPER/CORN/WEAT/SOYB/EMB/BWX/FXA/FXB/FXC)",
    )
    p.add_argument(
        "--fx-carry",
        action="store_true",
        dest="use_fx_carry",
        help="use FRED rate-differential carry for FX ETFs instead of dividend yield",
    )
    p.add_argument(
        "--diversifier-pack",
        action="store_true",
        dest="diversifier_pack",
        help="add BNDX/PFF/AMLP/MUB/EMLC cross-asset diversifiers (watch leverage)",
    )
    p.add_argument(
        "--rate-pack",
        action="store_true",
        dest="rate_pack",
        help="add BNDX/MUB international/rate bond diversifiers (watch leverage)",
    )
    p.add_argument("--monitor", action="store_true", help="rolling 1y Sharpe edge-decay monitor")
    p.add_argument("--placebo", type=int, default=12, help="random-walk placebo runs")
    p.add_argument(
        "--garch-vol",
        action="store_true",
        dest="use_garch_vol",
        help="blend in a GARCH(1,1) forward-vol estimate (requires `pip install -e .[garch]`)",
    )
    p.add_argument(
        "--garch-weight",
        type=float,
        default=0.0,
        dest="garch_weight",
        help="weight on GARCH vol (0 = pure EWMA blend, 1 = pure GARCH)",
    )
    p.add_argument("--garch-min-history", type=int, default=252, dest="garch_min_history")
    p.add_argument("--garch-refit-step", type=int, default=63, dest="garch_refit_step")
    p.add_argument("--garch-horizon", type=int, default=1, dest="garch_horizon")
    p.add_argument(
        "--max-gross",
        type=float,
        default=None,
        dest="max_gross",
        help="gross notional exposure cap as a multiple of capital (e.g. 3.0). None = uncapped.",
    )
    p.add_argument(
        "--financing-rate",
        type=float,
        default=0.0,
        dest="financing_rate",
        help="annual financing spread charged on gross notional above --financing-threshold (e.g. 0.015)",
    )
    p.add_argument(
        "--financing-threshold",
        type=float,
        default=1.0,
        dest="financing_threshold",
        help="multiple of capital below which no financing cost is charged (default 1.0)",
    )
    p.add_argument(
        "--max-annual-financing-cost",
        type=float,
        default=None,
        dest="max_annual_financing_cost",
        help="hard cap on annual financing cost as a fraction of capital (e.g. 0.02 = 2%%/year)",
    )
    p.add_argument(
        "--drawdown-control",
        action="store_true",
        dest="use_drawdown_control",
        help="opt-in drawdown-state control: de-risk on deep drawdowns, re-risk on recovery",
    )
    p.add_argument(
        "--drawdown-threshold",
        type=float,
        default=0.10,
        dest="drawdown_threshold",
        help="drawdown level that triggers de-risk (default 0.10 = 10%%)",
    )
    p.add_argument(
        "--drawdown-scale",
        type=float,
        default=0.50,
        dest="drawdown_scale",
        help="position scaling when drawdown control is active (default 0.50 = 50%%)",
    )
    p.add_argument(
        "--drawdown-recovery",
        type=float,
        default=0.05,
        dest="drawdown_recovery",
        help="drawdown recovery level that re-risks positions (default 0.05 = 5%%)",
    )
    p.add_argument(
        "--trend-strength-filter",
        action="store_true",
        dest="use_trend_strength_filter",
        help="opt-in trend-strength filter: de-gear when average absolute forecast is weak",
    )
    p.add_argument(
        "--trend-strength-window",
        type=int,
        default=63,
        dest="trend_strength_window",
        help="lookback window for trend-strength historical benchmark (default 63)",
    )
    p.add_argument(
        "--trend-strength-threshold",
        type=float,
        default=0.25,
        dest="trend_strength_threshold",
        help="bottom percentile of recent forecast strength that triggers de-risk (default 0.25)",
    )
    p.add_argument(
        "--trend-strength-scale",
        type=float,
        default=0.70,
        dest="trend_strength_scale",
        help="position scaling when trend strength is weak (default 0.70 = 70%%)",
    )
    args = p.parse_args(argv)
    if args.ship_candidate:
        args.expanded_universe = True
        args.regime_overlay = True
        args.buffer = 0.30
        args.regime_smooth = 5
    np.seterr(all="ignore")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
