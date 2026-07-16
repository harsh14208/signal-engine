# Financing, leverage, and honest comparison of strategy additions

## The problem

The engine targets a fixed volatility (typically 20% annualised). Low-volatility instruments — especially bond ETFs — must be held at **several times capital** to reach that target. Without charging for the leverage required to do that, a bond or diversifier pack can look like a "free" Sharpe improvement compared with adding an unlevered equity ETF.

That is not an apples-to-apples comparison. The financing model fixes it.

## The model

Three parameters control leverage cost:

- `financing_rate` — annual spread charged on the leveraged portion of gross notional (e.g., `0.01` = 1%).
- `financing_threshold` — gross-notional multiple below which no financing is charged (default `1.0`, i.e., the first 1× of capital is "free").
- `max_annual_financing_cost` — optional hard cap on annual financing drag as a fraction of capital (e.g., `0.02` = 2%/year). Positions are scaled down if the projected annual cost would exceed this.

Daily financing cost is:

```
financing = max(gross_notional / capital - financing_threshold, 0)
            * financing_rate / BUSINESS_DAYS_YEAR
```

This cost is subtracted from daily net returns, so it flows into Sharpe, drawdown, Calmar, and every downstream metric.

## CLI usage

```bash
python -m signal_engine ... --max-gross 3.0 --financing-rate 0.01 --financing-threshold 1.0

# Hard cap on financing drag instead of (or in addition to) a gross cap:
python -m signal_engine ... --financing-rate 0.01 --max-annual-financing-cost 0.015
```

A warning is emitted if `--max-gross > 1.0` is used with `--financing-rate=0`, because that combination overstates levered strategies.

Default behaviour is unchanged: `financing_rate=0.0`, so existing results are reproducible unless the flags are explicitly set.

## What the numbers show

All variants below use `--max-gross 3.0` and the 5-fold purged walk-forward from `scripts/eval_options_financing.py`.

| Variant | Net SR (no financing) | Net SR (1% financing) | WF OOS SR (1% financing) |
|---|---:|---:|---:|
| baseline (no cap) | 0.690 | — | 0.633 |
| baseline (3× cap) | 0.646 | 0.539 | 0.427 |
| + network momentum | 0.650 | 0.543 | 0.422 |
| + QQQ | 0.628 | 0.522 | 0.474 |
| + semis | 0.646 | 0.550 | **0.493** |
| + diversifier pack | 0.639 | 0.510 | 0.430 |
| + rate pack | 0.641 | 0.513 | 0.418 |
| + diversifier + COT + carry | 0.692 | 0.564 | 0.458 |
| weight: corr-cluster | 0.623 | 0.502 | 0.406 |
| weight: sharpe | 0.629 | 0.515 | 0.473 |

A 1% annual financing spread reduces Sharpe by roughly **0.10–0.13** across the board. Because the capped bond/diversifier packs still run ~2.7× average gross notional, the cost is material.

## Wide sensitivity grid

`scripts/eval_options_sensitivity_grid.py` runs the top variants across caps `{None, 4×, 3×, 2.5×, 2×}` and financing spreads `{0%, 0.5%, 1%, 1.5%, 2%}`. Walk-forward OOS Sharpe at selected points:

| Variant | cap=3×, 0% | cap=3×, 1% | cap=3×, 2% | cap=2.5×, 1% | cap=2×, 1% |
|---|---:|---:|---:|---:|---:|
| baseline | 0.548 | 0.427 | 0.305 | 0.385 | 0.355 |
| + semis | 0.600 | **0.493** | 0.385 | **0.487** | **0.456** |
| + QQQ | 0.593 | 0.474 | 0.355 | 0.451 | 0.407 |
| + diversifier pack | 0.581 | 0.430 | 0.280 | 0.399 | 0.386 |
| + rate pack | 0.566 | 0.418 | 0.271 | 0.398 | 0.397 |
| weight: corr-cluster | 0.543 | 0.406 | 0.268 | 0.378 | 0.374 |
| weight: sharpe | 0.598 | 0.473 | 0.348 | 0.447 | 0.419 |

Observations:

- `+semis` is the most robust across all realistic financing/cap combinations.
- `+QQQ` and `weight: sharpe` are also resilient.
- Bond packs (`+diversifier pack`, `+rate pack`) and `corr-cluster` degrade fastest as financing rises.
- Uncapped (`NaN`) results show very high full-sample Sharpe for bond packs, but they rely on 4–6×+ gross notional and collapse once any cap or financing is applied.

## Interpretation

- **Without financing**, the top results are levered bond/diversifier packs and corr-cluster weighting. They look like clear winners.
- **With financing**, those leaders fall back into the pack. `+semis` and `+QQQ` become competitive because they are already equity-like and do not depend on hidden leverage.
- The **corr-cluster** weighting collapses the most (OOS 0.543 → 0.406 at 3×/1%), suggesting its edge was largely an artifact of concentrating risk in low-vol, highly levered instruments.
- The best single financed OOS result is `+diversifier + COT + carry` (0.458), but its IS/OOS gap widens to +0.21.

## Practical implication

Do **not** promote bond-pack or corr-cluster variants to default without also setting `--financing-rate`. The already-shipped flags (`--semis`, `--qqq`, `--network-momentum`) are safer because:

1. They do not rely on hidden leverage.
2. Their rankings improve on a like-for-like cost basis.
3. They keep trade frequency and gross exposure in line with the baseline.

## Cross-check against the six research/patent optimizations

`scripts/eval_optimizations.py` re-ran the new opt-in levers (calibration smoothing,
drawdown control, trend-strength filter, network momentum) under the same 3× cap +
1% financing bar. None beat the baseline on walk-forward OOS Sharpe:

| Configuration | Net SR | WF OOS SR | Max DD |
|---|---:|---:|---:|
| Baseline | 0.539 | 0.427 | −33.2% |
| + drawdown control | 0.510 | 0.417 | −26.2% |
| + trend-strength filter | 0.536 | 0.428 | −31.4% |
| + calibration smooth | 0.539 | 0.427 | −33.2% |
| + drawdown + trend | 0.496 | 0.444 | −26.2% |

This confirms that the financing model is the dominant real-world constraint;
overlay refinements are secondary until they can clear the same honest bar.

## Files and scripts

- Implementation: `signal_engine/config.py`, `signal_engine/cli.py`, `signal_engine/backtest.py`, `signal_engine/live.py`
- Evaluation scripts: `scripts/eval_options_financing.py`, `scripts/eval_options_sensitivity_grid.py`
- Live target generator: `scripts/generate_targets.py` (supports `--max-gross`, `--financing-rate`, `--financing-threshold`, `--max-annual-financing-cost`)
- Raw results: `data/options_evaluation_financing.json`, `data/options_evaluation_sensitivity_grid.json`
- This write-up: `docs/FINANCING_AND_LEVERAGE.md`
