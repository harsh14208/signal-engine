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
| **H4 beats_buy_hold** (hard) | CAGR/MaxDD/Sharpe vs. a trivial SPY buy-and-hold, on a CAGR > MaxDD > Sharpe > Calmar priority (2026-07-22) |
| **R1 cpcv_robust** | CPCV OOS 5th-pct > 0 and < 25% of paths below zero |
| **R2 walk_forward_ok** | walk-forward mean OOS Sharpe > 0 and IS−OOS gap < 0.5 |
| **R3 cost_headroom** | break-even cost ≥ 2× the assumed cost |

Any hard failure ⇒ **FAIL**. Hard-pass but a robustness failure ⇒ **CONDITIONAL**.
All pass ⇒ **PASS**. ENB and the live forward track are reported as context.

### Verdict on real cached data (2026-07-09): ❌ H3 FAIL after correcting the trial count

The original run reported a razor-thin PASS:

```
H1 clears_noise:    net 0.71 vs noise-floor 0.40   ✅
H2 edge_real:       bootstrap P5 0.38 > 0          ✅
H3 passes_deflated: net 0.71 vs deflated-max 0.69  ✅  ← clears by 0.02
R1 cpcv_robust:     OOS P5 0.48, 0% paths<0        ✅
R2 walk_forward_ok: OOS 0.62, gap 0.16             ✅
R3 cost_headroom:   net Sharpe positive past 25bps ✅
effective bets 12.9 / 19
```

**Updated read (2026-07-15):** the trial counter was undercounting.
`trial_registry.jsonl` held 15 fingerprints while the actual search log
`experiments.jsonl` held 153 raw config hashes. `validation.honest_n_trials()` now
unions both sources and deduplicates by effective Config, giving **141 distinct
strategies searched**. At that count the Deflated-Sharpe bar rises to ≈**0.72**, and
the baseline net Sharpe of **0.69 fails H3**.

| n_trials | Deflated max | Baseline 0.69 passes? |
|---:|---:|:---|
| 15 (stale registry) | 0.53 | ✅ |
| 100 (prior floor) | 0.69 | ❌ |
| 141 (honest count) | 0.72 | ❌ |

This is the parent engine's failure signature reproduced in real time: every null
backtest raises the bar on the same edge. The gate is doing its job — it makes the
fragility visible instead of hiding it.

**Conclusion:** backtest evidence alone can no longer certify the edge. The default
engine remains the best backtested candidate, but promotion now depends on the
accruing forward track (`data/live_returns.csv`) via the champion/challenger
lifecycle, not on further backtests.

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

## What's next
Phase 2 (provider reliability, market calendar, point-in-time feature store) and
Phase 3 (replay-based decision-level drift detection) are **shipped** — see
`docs/PHASE2_DATA.md` / `docs/PHASE3_REPLAY.md`. Phase 4 (HRP + PCA risk) remains
unbuilt, justified only if the edge holds on real-futures data and a live track.

### Research-only levers (not subjected to the full Phase 0 bar)

Six additional optimizations from a 2026 research sweep were implemented as
opt-in diagnostics: implementation-shortfall drift decomposition, warm-up/stateful-
restart parity, quartile edge-decay kill switch, calibration smoothing, drawdown-
state control, and trend-strength filter. A financed 3×-cap evaluation
(`scripts/eval_optimizations.py`) found that **none improves walk-forward OOS
Sharpe versus the baseline**, so they remain research flags. They were not run
through the full Phase 0 honesty battery because the first OOS check already failed;
running more tests would only increase false-discovery risk. See
`docs/OPTIMIZATIONS.md`.
