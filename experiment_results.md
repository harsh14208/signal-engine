# Experiment results — Tier 0 levers

Date: 2026-06-28 01:16 UTC
Data: real ETF-proxy prices (cache), 2007–2026.
Validation: 70/30 chronological OOS, random-walk placebo (n=12), Lo CI (n_trials=100), block-bootstrap.

## Summary table

| Run | Net SR | IS SR | OOS SR | IS−OOS gap | Ann vol | Max DD | Turnover | IDM | FDM | Placebo 95th | Deflated max | BB P5 | Div ratio |
|-----|-------:|------:|-------:|------------:|--------:|-------:|---------:|----:|----:|-------------:|-------------:|------:|----------:|
| baseline | 0.69 | 0.76 | 0.55 | +0.21 | 21.4% | -38.2% | 47.1x | 2.13 | 1.14 | 0.35 | 0.69 | 0.35 | 2.4x |
| carry_proxies | 0.71 | 0.80 | 0.52 | +0.28 | 21.5% | -38.6% | 47.0x | 2.13 | 1.30 | 0.35 | 0.69 | 0.37 | 2.4x |
| empirical_scalars | 0.69 | 0.77 | 0.50 | +0.27 | 21.5% | -41.4% | 45.0x | 2.13 | 1.13 | 0.40 | 0.69 | 0.35 | 2.4x |
| regime_overlay | 0.68 | 0.74 | 0.55 | +0.19 | 21.4% | -39.3% | 52.3x | 2.13 | 1.14 | 0.35 | 0.69 | 0.33 | 2.4x |
| carry+scalars | 0.73 | 0.80 | 0.57 | +0.24 | 21.4% | -38.0% | 44.4x | 2.13 | 1.29 | 0.40 | 0.69 | 0.39 | 2.4x |
| carry+regime | 0.70 | 0.77 | 0.54 | +0.24 | 21.4% | -40.2% | 52.4x | 2.13 | 1.30 | 0.35 | 0.69 | 0.35 | 2.4x |
| scalars+regime | 0.68 | 0.74 | 0.55 | +0.19 | 21.3% | -37.1% | 49.0x | 2.13 | 1.13 | 0.40 | 0.69 | 0.33 | 2.4x |
| carry+scalars+regime | 0.73 | 0.78 | 0.62 | +0.16 | 21.2% | -34.5% | 48.5x | 2.13 | 1.29 | 0.40 | 0.69 | 0.38 | 2.4x |
| expanded_universe | 0.70 | 0.75 | 0.60 | +0.15 | 21.4% | -40.2% | 60.6x | 1.98 | 1.14 | 0.56 | 0.69 | 0.36 | 2.9x |
| expanded+carry | 0.70 | 0.76 | 0.56 | +0.20 | 21.4% | -41.5% | 61.1x | 1.98 | 1.29 | 0.56 | 0.69 | 0.36 | 3.0x |
| expanded+regime | 0.73 | 0.74 | 0.71 | +0.03 | 21.1% | -36.1% | 67.0x | 1.98 | 1.14 | 0.56 | 0.69 | 0.39 | 2.9x |
| ship_candidate | 0.74 | 0.75 | 0.72 | +0.03 | 21.2% | -36.0% | 62.7x | 1.98 | 1.14 | 0.56 | 0.69 | 0.40 | 2.9x |
| ship+vix_term | 0.72 | 0.74 | 0.68 | +0.06 | 21.2% | -35.2% | 63.8x | 1.98 | 1.14 | 0.56 | 0.69 | 0.40 | 2.9x |
| ship+credit | 0.74 | 0.76 | 0.72 | +0.04 | 21.2% | -35.6% | 62.9x | 1.98 | 1.14 | 0.56 | 0.69 | 0.40 | 2.9x |

## Notes

- All runs use expanding-window calibration for weights/IDM/FDM (no full-sample leak).
- Baseline: Net SR 0.69, OOS SR 0.55, gap +0.21.
- Best net Sharpe: **ship+credit** → Net 0.74, OOS 0.72, gap +0.04, turnover 62.9x.
- `carry_proxies`: large gap (+0.28) — left opt-in.
- `empirical_scalars`: large gap (+0.27) — left opt-in.
- Default `buffer_fraction` is now 30% (up from 10%). This single parameter change raised baseline OOS Sharpe from 0.51 to 0.55 and cut turnover from ~60x to ~47x.
- `--ship-candidate` is available as an opt-in preset: expanded universe + regime overlay + 30% buffer + regime smooth=5. It delivers Net 0.74 / OOS 0.72 / gap +0.03 / turnover ~63x.
- Free macro overlays tested on top of `--ship-candidate`:
  - `ship+vix_term` (VIX 9D/spot + 3M/term structure): net SR 0.72, OOS 0.68, gap +0.06, turnover 63.8x — **worse** than ship alone.
  - `ship+credit` (Baa-10Y credit risk premium, tuned upper=1.3 / lower=0.7 / lookback=756): net SR 0.74, OOS 0.72, gap +0.04, turnover 62.9x — **statistically tied** with ship alone. The threshold was selected by a small grid search, so the tie is likely overfit.
- Conclusion: neither free-data overlay beats the ship candidate on honest OOS turnover/gap grounds. Both are kept as **opt-in CLI flags** (`--vix-term-overlay`, `--credit-overlay`) for further research, but are **not promoted to default**.
