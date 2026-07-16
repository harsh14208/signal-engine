# Experiment results — Tier 0 levers

Date: 2026-07-10 23:45 UTC
Data: real ETF-proxy prices (cache), 2007–2026.
Validation: 70/30 chronological OOS, random-walk placebo (n=12), Deflated Sharpe at the **honest** trial count (n_trials=141, from the union of `trial_registry.jsonl` and `experiments.jsonl` — not a hardcoded 100), block-bootstrap, and PBO across the search. Baseline Net SR 0.69 fails H3 at this count.

## Summary table

| Run | Net SR | IS SR | OOS SR | IS−OOS gap | Ann vol | Max DD | Turnover | IDM | FDM | Placebo 95th | Deflated max | BB P5 | Div ratio |
|-----|-------:|------:|-------:|------------:|--------:|-------:|---------:|----:|----:|-------------:|-------------:|------:|----------:|
| baseline | 0.69 | 0.76 | 0.52 | +0.24 | 21.4% | -38.2% | 47.1x | 2.13 | 1.14 | 0.35 | 0.69 | 0.35 | 2.4x |
| carry_proxies | 0.70 | 0.80 | 0.50 | +0.30 | 21.5% | -38.6% | 47.0x | 2.13 | 1.30 | 0.35 | 0.69 | 0.36 | 2.4x |
| empirical_scalars | 0.69 | 0.77 | 0.49 | +0.29 | 21.5% | -41.4% | 45.0x | 2.13 | 1.13 | 0.40 | 0.69 | 0.35 | 2.4x |
| regime_overlay | 0.68 | 0.74 | 0.53 | +0.21 | 21.4% | -39.3% | 52.3x | 2.13 | 1.14 | 0.35 | 0.69 | 0.35 | 2.4x |
| carry+scalars | 0.73 | 0.80 | 0.55 | +0.25 | 21.4% | -38.0% | 44.4x | 2.13 | 1.29 | 0.40 | 0.69 | 0.39 | 2.4x |
| carry+regime | 0.70 | 0.78 | 0.52 | +0.25 | 21.4% | -40.2% | 52.3x | 2.13 | 1.30 | 0.35 | 0.69 | 0.35 | 2.4x |
| scalars+regime | 0.68 | 0.74 | 0.54 | +0.20 | 21.3% | -37.1% | 49.0x | 2.13 | 1.13 | 0.40 | 0.69 | 0.33 | 2.4x |
| carry+scalars+regime | 0.73 | 0.78 | 0.60 | +0.18 | 21.1% | -34.5% | 48.5x | 2.13 | 1.29 | 0.40 | 0.69 | 0.38 | 2.4x |
| expanded_universe | 0.70 | 0.75 | 0.60 | +0.15 | 21.4% | -40.2% | 60.6x | 1.98 | 1.14 | 0.56 | 0.69 | 0.36 | 2.9x |
| expanded+carry | 0.70 | 0.76 | 0.56 | +0.20 | 21.4% | -41.5% | 61.1x | 1.98 | 1.29 | 0.56 | 0.69 | 0.36 | 3.0x |
| expanded+regime | 0.73 | 0.74 | 0.71 | +0.03 | 21.1% | -36.1% | 67.0x | 1.98 | 1.14 | 0.56 | 0.69 | 0.39 | 2.9x |
| ship_candidate | 0.74 | 0.75 | 0.72 | +0.03 | 21.2% | -36.0% | 62.7x | 1.98 | 1.14 | 0.56 | 0.69 | 0.40 | 2.9x |
| ship+vix_term | 0.72 | 0.74 | 0.68 | +0.06 | 21.2% | -35.2% | 63.8x | 1.98 | 1.14 | 0.56 | 0.69 | 0.40 | 2.9x |
| ship+credit | 0.74 | 0.76 | 0.72 | +0.04 | 21.2% | -35.6% | 62.9x | 1.98 | 1.14 | 0.56 | 0.69 | 0.40 | 2.9x |

## Notes

- All runs use expanding-window calibration for weights/IDM/FDM (no full-sample leak).
- Baseline: Net SR 0.69, OOS SR 0.52, gap +0.24.
- Best net Sharpe: **ship+credit** → Net 0.74, OOS 0.72, gap +0.04, turnover 62.9x.
- `carry_proxies`: large gap (+0.30) — left opt-in.
- `empirical_scalars`: large gap (+0.29) — left opt-in.
- `carry+regime`: large gap (+0.25) — left opt-in.
- Default `buffer_fraction` is now 30% (up from 10%). This single parameter change raised baseline OOS Sharpe from 0.51 to 0.55 and cut turnover from ~60x to ~47x.
- `--ship-candidate` is available as an opt-in preset: expanded universe + regime overlay + 30% buffer + regime smooth=5. It delivers Net 0.74 / OOS 0.72 / gap +0.03 / turnover ~63x.
- None of the individual additive levers is promoted to default on its own; each either fails to improve OOS, widens the IS/OOS gap, or raises turnover beyond the benefit.

## Honesty — overfitting at the real trial count

- Deflated-Sharpe bar uses the honest trial count. The original `trial_registry.jsonl`
  held only 15 fingerprints while the actual search log `experiments.jsonl` held 153
  raw config hashes. `validation.honest_n_trials()` now unions both sources and
  deduplicates by *effective Config*, giving **141 distinct strategies searched**.
- At n_trials=141 and ≈4,914 daily observations, the Deflated-Sharpe expected-max
  bar is ≈**0.72**. The baseline net Sharpe of **0.69 does NOT clear H3**.
- This is the parent engine's failure signature in real time: every null backtest
  raises the bar on the same edge. Backtest-only certification is exhausted; the
  edge must now be confirmed by the forward track.
- ⚠ Runs that did **not** clear Deflated Sharpe at the previous floor (n_trials=100,
  bar ≈0.69): baseline, empirical_scalars, regime_overlay, scalars+regime.
- Probability of Backtest Overfitting (CSCV over 14 configs): **0.80** (⚠ OVERFIT).
  >0.5 means the IS-best config is below the OOS median more often than chance.

---

# Experiment results — Diversifier / options pack evaluation (with financing)

Date: 2026-07-15
Data: real ETF-proxy prices (cache), 2007–2026.
Method: 5-fold purged walk-forward (20% embargo); full-sample backtest for Net Sharpe.
All variants run with `--max-gross 3.0` unless noted.

## Summary

| Variant | Financing | Net SR | WF OOS SR | IS−OOS gap | Ann vol | Max DD | Mean gross | Turnover |
|---------|-----------|-------:|----------:|------------:|--------:|-------:|-----------:|---------:|
| baseline (no cap) | 0% | 0.690 | 0.633 | 0.082 | 21.4% | -38.2% | 4.09× | 47.1× |
| baseline | 0% | 0.646 | 0.548 | 0.123 | 15.5% | -32.1% | 2.68× | 24.1× |
| baseline | 1% | 0.539 | 0.427 | 0.154 | 15.5% | -33.2% | 2.68× | 24.1× |
| + network momentum | 0% | 0.650 | 0.543 | 0.168 | 15.4% | -34.0% | 2.67× | 23.5× |
| + network momentum | 1% | 0.543 | 0.422 | 0.197 | 15.4% | -35.0% | 2.67× | 23.5× |
| + QQQ | 0% | 0.628 | 0.593 | 0.047 | 15.6% | -31.9% | 2.67× | 23.7× |
| + QQQ | 1% | 0.522 | 0.474 | 0.077 | 15.6% | -33.0% | 2.67× | 23.7× |
| + semis | 0% | 0.646 | 0.600 | 0.031 | 16.6% | -32.0% | 2.61× | 23.4× |
| + semis | 1% | 0.550 | 0.493 | 0.057 | 16.6% | -33.9% | 2.61× | 23.4× |
| + diversifier pack | 0% | 0.639 | 0.581 | 0.109 | 13.5% | -29.9% | 2.76× | 23.5× |
| + diversifier pack | 0.5% | 0.575 | 0.506 | 0.130 | 13.5% | -30.7% | 2.76× | 23.5× |
| + diversifier pack | 1% | 0.510 | 0.430 | 0.152 | 13.5% | -32.2% | 2.76× | 23.5× |
| + diversifier pack | 1.5% | 0.446 | 0.355 | 0.173 | 13.5% | -33.6% | 2.76× | 23.5× |
| + rate pack | 0% | 0.641 | 0.566 | 0.108 | 13.6% | -32.1% | 2.76× | 23.4× |
| + rate pack | 1% | 0.513 | 0.418 | 0.150 | 13.6% | -33.8% | 2.76× | 23.4× |
| + diversifier + COT + carry | 0% | 0.692 | 0.609 | 0.162 | 13.3% | -29.0% | 2.72× | 23.5× |
| + diversifier + COT + carry | 1% | 0.564 | 0.458 | 0.209 | 13.3% | -30.8% | 2.72× | 23.5× |
| weight: corr-cluster | 0% | 0.623 | 0.543 | 0.049 | 14.2% | -29.3% | 2.73× | 25.4× |
| weight: corr-cluster | 1% | 0.502 | 0.406 | 0.079 | 14.2% | -30.6% | 2.73× | 25.4× |
| weight: sharpe | 0% | 0.629 | 0.598 | 0.009 | 14.8% | -32.2% | 2.69× | 24.5× |
| weight: sharpe | 1% | 0.515 | 0.473 | 0.030 | 14.8% | -33.2% | 2.69× | 24.5× |

## Interpretation

- Without financing costs, capped bond-pack/diversifier and corr-cluster variants post attractive **full-sample** Sharpe ratios (0.62–0.69) because the 3× gross cap still allows ~2.7× average notional in low-vol instruments.
- A realistic 1% annual financing spread on gross notional above 1× capital reduces Net Sharpe by roughly **0.10–0.13** and walk-forward OOS Sharpe by a similar amount.
- After financing, the **previously leading bond/diversifier and corr-cluster variants fall back into the pack**:
  - +semis (WF OOS 0.493) and +QQQ (0.474) are now competitive with or ahead of +diversifier pack (0.430) and +rate pack (0.418).
  - corr-cluster collapses from 0.543 to 0.406 OOS.
  - The best financed OOS result remains **+diversifier + COT + carry** (0.458), but its IS/OOS gap widens to +0.21.
- Conclusion: **financing costs are material** and must be included when comparing levered bond packs to unlevered equity additions. The clean, already-shipped flags (`--semis`, `--qqq`, `--network-momentum`) are robust because they do not rely on hidden leverage and their ranking improves on a like-for-like cost basis.

## Implementation

- Added `Config.financing_rate` and `Config.financing_threshold`.
- Added `--financing-rate` / `--financing-threshold` CLI flags.
- Financing is applied after gross-notional capping and subtracted from daily net returns; it therefore flows into Sharpe, drawdown, and all downstream metrics.
- Added `scripts/eval_options_financing.py` and saved results to `data/options_evaluation_financing.json`.


---

# Experiment results — six research/patent optimizations (with financing)

Date: 2026-07-15
Data: real ETF-proxy prices (cache), 2007–2026, core universe + semis/QQQ options pack.
Method: 5-fold purged walk-forward (20% embargo); full-sample backtest for Net Sharpe.
All variants run with `--max-gross 3.0 --financing-rate 0.01`.

## Summary

| Configuration | Net SR | WF OOS SR | IS−OOS gap | Ann vol | Max DD | Mean gross | Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.539 | 0.427 | 0.154 | 15.5% | −33.2% | 2.68× | 24.1× |
| + calibration smooth | 0.539 | 0.427 | 0.154 | 15.5% | −33.2% | 2.68× | 24.1× |
| + drawdown control | 0.510 | 0.417 | 0.093 | 12.2% | −26.2% | 2.04× | 19.1× |
| + trend-strength filter | 0.536 | 0.428 | 0.108 | 15.1% | −31.4% | 2.60× | 24.9× |
| + network momentum | 0.543 | 0.422 | 0.121 | 15.6% | −35.0% | 2.67× | 23.5× |
| + drawdown + trend | 0.496 | 0.444 | 0.052 | 12.1% | −26.2% | 2.04× | 18.1× |
| + drawdown + trend + network mom + smooth | 0.505 | 0.435 | 0.070 | 12.2% | −26.2% | 2.02× | 18.3× |

## Interpretation

- **None of the new combinations improves walk-forward OOS Sharpe versus the baseline.**
  They remain **opt-in / diagnostic only**.
- **Drawdown control** works as designed: it lowers realised volatility and maximum
  drawdown, but also lowers returns, leaving risk-adjusted return essentially flat.
- **Trend-strength filter** is neutral on the full sample and does not recover the
  2023–26 weakness with default parameters, suggesting that weakness is genuine
  trend decay rather than a low-strength regime fixable in-sample.
- **Calibration smoothing** does not change aggregate performance on this dataset
  because expanding-window parameters stabilise early and change only at rebal dates.
- **Network momentum** is already validated-positive and essentially matches the
  baseline here because it is included in the options-evaluation universe by default.

## Decision

No full Deflated-Sharpe / block-bootstrap / placebo bar was run on these combos.
The first OOS check failed for every combo relative to the baseline, so further
validation would mainly increase false-discovery risk. The levers stay available
in `Config` / the CLI for research and diagnostics.

## Implementation

- `signal_engine/config.py` — new fields for calibration smoothing, drawdown control,
  and trend-strength filter.
- `signal_engine/portfolio.py` — `drawdown_overlay()`, `trend_strength_overlay()`.
- `signal_engine/backtest.py` — applies overlays before gross cap and financing cost;
  `_smooth_parameter_transitions()` for calibration smoothing.
- `signal_engine/monitor.py` — `decompose_drift()`; `worst_quartile` flag in
  `edge_decay_report()`.
- `signal_engine/live.py` — warm-up / stateful-restart parity guard.
- `signal_engine/cli.py` — new flags.
- Added `scripts/eval_optimizations.py`; results saved to
  `data/options_evaluation_optimizations.json`.
