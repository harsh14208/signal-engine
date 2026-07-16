# Phase 0 (edge gate) + Phase 1 (realized friction)

The go/no-go gate that decides whether platform investment is justified, plus the
tooling that makes the cost assumption empirical instead of hardcoded.

## Phase 0 — the edge gate

```bash
python scripts/validate_edge.py --source cache   # real data, not synthetic
```

Exit codes: `0=PASS`, `1=CONDITIONAL`, `2=FAIL`, `3=error`. Pre-registered gates:

| Gate | Criterion |
|---|---|
| **H1 clears_noise** (hard) | net Sharpe > placebo 95th-pct noise floor |
| **H2 edge_real** (hard) | block-bootstrap 5th-pct Sharpe > 0 |
| **H3 passes_deflated** (hard) | Deflated Sharpe passes at the honest trial count |
| **R1 cpcv_robust** | CPCV OOS 5th-pct > 0 and < 25% of paths below zero |
| **R2 walk_forward_ok** | walk-forward mean OOS Sharpe > 0 and IS−OOS gap < 0.5 |
| **R3 cost_headroom** | break-even cost ≥ 2× the assumed cost |

Any hard failure ⇒ **FAIL**. Hard-pass but a robustness failure ⇒ **CONDITIONAL**.
All pass ⇒ **PASS**. ENB and the live forward track are reported as context.

### Verdict on real cached data (2026-07-09): ✅ PASS — but H3 is razor-thin

```
H1 clears_noise:    net 0.71 vs noise-floor 0.40   ✅
H2 edge_real:       bootstrap P5 0.38 > 0          ✅
H3 passes_deflated: net 0.71 vs deflated-max 0.69  ✅  ← clears by 0.02
R1 cpcv_robust:     OOS P5 0.48, 0% paths<0        ✅
R2 walk_forward_ok: OOS 0.62, gap 0.16             ✅
R3 cost_headroom:   net Sharpe positive past 25bps ✅
effective bets 12.9 / 19
```

**Read this honestly:** the edge passes, but H3 (0.71 vs 0.69 at 100 trials) is the
exact margin the parent engine *failed*. Two caveats gate the "PASS":
1. It is measured on **ETF proxies with proxy carry**, not real futures.
2. The deflation used a floored 100-trial prior. As the trial registry grows
   (every `validate_edge` run and experiment registers one), the honest trial
   count rises and the deflated-max threshold with it — this could flip H3 to FAIL.
   That is the point: the gate makes the fragility visible instead of hiding it.

**Conclusion:** enough of a signal to justify Phase 1+ investment, *conditional on*
(a) confirming on real-futures data and (b) accruing a live forward track. Do not
treat PASS as "the edge is proven."

## Phase 1 — realized-friction calibration

The whole *net* Sharpe rests on an assumed 1.5 bps/side. `signal_engine/friction.py`:

- **Measure it** — `realized_friction(fills)` backs out per-side slippage +
  commission (decision price vs fill price) per instrument, from real broker fills.
- **Stress it** — `cost_break_even(prices, cfg)` / `net_of_friction_curve(...)`
  re-run the backtest across a cost grid and report the round-trip cost at which
  the edge dies. On real data the net edge stays positive well past 25 bps — large
  headroom over 1.5 bps (this is what earns R3).
- **Use it** — `write_calibration(friction)` persists measured per-symbol costs to
  `data/friction_calibration.json`; `Config(cost_scheme="calibrated")` then prices
  the backtest off real fills, falling back to the flat assumption per uncalibrated
  symbol.

Ported method mirrors TRS §104b: *measure* the real cost and stress the assumption;
never silently change the constant.

## What's next (unbuilt)
Phase 2 (provider reliability, market calendar, point-in-time feature store),
Phase 3 (replay-based decision-level drift detection), Phase 4 (HRP + PCA risk) —
all justified only if the edge holds on real-futures data and a live track.

### Research-only levers (not subjected to the full Phase 0 bar)

Six additional optimizations from the research/patent sweep were implemented as
opt-in diagnostics: implementation-shortfall drift decomposition, warm-up/stateful-
restart parity, quartile edge-decay kill switch, calibration smoothing, drawdown-
state control, and trend-strength filter. A financed 3×-cap evaluation
(`scripts/eval_optimizations.py`) found that **none improves walk-forward OOS
Sharpe versus the baseline**, so they remain research flags. They were not run
through the full Phase 0 honesty battery because the first OOS check already failed;
running more tests would only increase false-discovery risk.
