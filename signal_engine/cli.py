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

from .backtest import run_backtest
from .carry_data import build_carry_panel
from .config import Config
from .data import load_prices, synthetic_carry
from .diagnostics import cost_buffer_frontier, per_instrument_attribution, vix_regime_split
from .experiments import count_experiments, log_experiment
from .macro import (
    credit_overlay,
    load_credit_spread,
    load_vix,
    load_vix_term_structure,
    regime_overlay,
    vix_term_overlay,
)
from .markets import symbols
from .metrics import sharpe
from .report import full_report
from .validation import block_bootstrap_sharpe, lo_sharpe_ci, placebo_sharpes, purged_walk_forward


def build_config(args) -> Config:
    return Config(
        capital=args.capital,
        vol_target=args.vol_target,
        use_breakout=not args.no_breakout,
        use_carry=args.carry,
        use_carry_proxies=args.carry_proxies,
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


def _print_walk_forward(prices, cfg, n_splits: int) -> None:
    wf = purged_walk_forward(prices, cfg, n_splits=n_splits)
    print("\n## Walk-forward / purged CV\n")
    if wf.get("insufficient"):
        print("- Insufficient data for walk-forward analysis.")
        return
    print(f"- Folds: {wf['n_folds']}  mean IS Sharpe: **{wf['mean_is_sharpe']:.2f}**")
    print(f"- mean OOS Sharpe: **{wf['mean_oos_sharpe']:.2f}**  mean gap: **{wf['mean_gap']:+.2f}**")
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
        vix = load_vix(prices.index.min().strftime("%Y-%m-%d"), prices.index.max().strftime("%Y-%m-%d"))
        split = vix_regime_split(result.daily_returns, vix)
        print(f"- Median VIX: {split['median_vix']:.1f}")
        for label, stats in [("High VIX", split["high_vix"]), ("Low VIX", split["low_vix"])]:
            print(
                f"- {label}: n={stats['n_days']}, Sharpe={stats['sharpe']:.2f}, "
                f"vol={stats['ann_vol']:.1%}, ret={stats['ann_return']:.1%}, MaxDD={stats['max_dd']:.1%}"
            )
    except Exception as exc:
        print(f"- Could not load VIX for regime split: {exc}")


def run(args) -> int:
    cfg = build_config(args)
    expanded = cfg.use_expanded_universe
    syms = symbols(expanded=expanded)
    cache_tag = "expanded" if expanded else "universe"
    prices = load_prices(syms, start=args.start, end=args.end, source=args.source, cache_tag=cache_tag)
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
        vix = load_vix(prices.index.min().strftime("%Y-%m-%d"), prices.index.max().strftime("%Y-%m-%d"))
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
        spread = load_credit_spread(
            prices.index.min().strftime("%Y-%m-%d"), prices.index.max().strftime("%Y-%m-%d")
        ).reindex(prices.index).ffill()
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

    result = run_backtest(prices, cfg, carry=carry, regime=regime)
    print(f"# signal-engine — {args.source} run\n")
    print(full_report(result))
    if args.oos:
        _print_oos(result, args.oos)
    if args.walk_forward:
        _print_walk_forward(prices, cfg, args.walk_forward)
    if args.diagnostics:
        _print_diagnostics(prices, result, cfg)
    if args.validate:
        _print_validation(result, cfg, args)

    # Log this experiment so the Deflated-Sharpe counter reflects real search.
    log_experiment(cfg, result)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="signal_engine", description=__doc__)
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
    p.add_argument("--buffer", type=float, default=0.10)
    p.add_argument("--no-breakout", action="store_true")
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
    p.add_argument("--vix-term-short-thresh", type=float, default=1.10, dest="vix_term_short_thresh")
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
    p.add_argument(
        "--corr-spike-span", type=int, default=60, dest="corr_spike_span"
    )
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
        help="validated best candidate: expanded universe + regime overlay + 30% buffer + regime smooth=5",
    )
    p.add_argument("--placebo", type=int, default=12, help="random-walk placebo runs")
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
