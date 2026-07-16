# signal-engine — final research summary

> Consolidated summary of all research results, experiments, and data artifacts as
> of 2026-07-15. See the linked docs for full methodology.

## 1. The thesis

A diversified systematic **trend + carry** engine (Carver / AHL style). The edge is
not any single instrument, but stacking ~20 weakly-correlated risk-adjusted bets and
scaling the basket to a constant volatility target via the **Instrument
Diversification Multiplier (IDM)**.

**Validated default config:** core 19 ETF proxies + realised-vol governor + 30%
no-trade buffer, trend-only.

---

## 2. Default-config results (real data, 2007–2026)

| Metric | Value |
|---|---|
| Net Sharpe | **0.69** |
| Gross Sharpe | 0.72 |
| Realised vol | 21.4% (vs 20% target) |
| Max drawdown | −38% |
| Calmar | 0.35 |
| Diversification ratio | 2.4× |
| Lo 95% CI | [0.23, 1.13] — SR=0 outside ✅ |
| Block-bootstrap P5 | 0.34 > 0 ✅ |
| Random-walk placebo | clears ✅ |
| Deflated Sharpe (honest trial count) | clears ✅ |
| 4-fold purged walk-forward mean OOS | **0.61**, gap +0.12 |

---

## 3. Validated-positive levers (cleared walk-forward OOS)

| Lever | Effect | Status |
|---|---|---|
| **Realised-vol governor** | Sharpe 0.54→0.65; MaxDD −49%→−36%; vol 25%→21% | **ON by default** |
| **30% no-trade buffer** | OOS Sharpe 0.51→0.55; turnover ~60×→~47× | **Default** |
| **COT positioning** (`--cot`) | Full 0.69→0.72; WF OOS 0.61→0.63 | **VALIDATED-POSITIVE**, opt-in |
| **Network momentum** (`--network-momentum`) | Full 0.69→0.73; WF OOS 0.63→0.67 | **VALIDATED-POSITIVE** |
| **Semis pack** (`--semis`) | WF OOS 0.63→0.69 | **VALIDATED-POSITIVE** |
| **QQQ** (`--qqq`) | WF OOS 0.63→0.68 | **VALIDATED-POSITIVE** |

---

## 4. Tested-but-opt-in levers (neutral or marginal)

| Lever | Result |
|---|---|
| Carry proxies (`--carry-proxies`) | Full 0.65→0.66, OOS flat 0.51 |
| Expanded universe (`--expanded-universe`) | Full 0.65→0.70, OOS 0.51→0.52, but gap widens and turnover jumps ~60×→~79× |
| Empirical scalars (`--empirical-scalars`) | Full flat, OOS 0.51→0.49 |
| Macro regime overlay (`--regime-overlay`) | Full flat, OOS 0.51→0.53, turnover up |
| Cross-sectional momentum (`--xsmom`) | Implemented, placebo-clear on synthetic |
| HMM regime overlay (`--hmm-regime-overlay`) | Inert |
| Real bond carry (`--real-bond-carry`) | Inert |
| 2s10s curve steepener (`--curve-steepener`) | Mixed walk-forward; turnover doubles |
| Corr-spike overlay (`--corr-spike`) | Exists, not promoted |

---

## 5. Confirmed dead-ends / parked

| Lever | Result |
|---|---|
| **Asset-class cluster weighting** | Sharpe 0.54→0.49, OOS 0.57→0.42 — singleton clusters overweight weak names |
| **VIX term-structure overlay** | Hurts ship-candidate OOS 0.72→0.68 |
| **Baa-10Y credit-spread overlay** | Tuned threshold ties ship-candidate; treated as no improvement |
| **GARCH(1,1) vol sizing** | OOS 0.55→0.49 |
| **S&P 500 cross-sectional momentum sleeve** | Net 0.69→0.62 |
| **VRP injection** | **Parked** — short-vol as tradable instrument detonates vol-targeting |

---

## 6. Financing / leverage-cost model

Without charging for leverage, low-vol bond packs look like free Sharpe. The
financing model fixes this.

**Parameters:**
- `financing_rate` — annual spread on levered gross notional above threshold
- `financing_threshold` — first 1× of capital is free (default)
- `max_annual_financing_cost` — optional hard cap
- `max_gross_notional` — gross-exposure cap

**Effect of 1% financing (3× cap):**
- Baseline Net Sharpe: 0.646 → **0.539**
- WF OOS Sharpe: 0.548 → **0.427**

**Best financed OOS results (3× cap, 1%):**

| Variant | Net SR | WF OOS SR |
|---|---:|---:|
| + semis | 0.550 | **0.493** |
| + QQQ | 0.522 | 0.474 |
| + diversifier + COT + carry | 0.564 | 0.458 |
| baseline | 0.539 | 0.427 |
| + rate pack | 0.513 | 0.418 |
| + diversifier pack | 0.510 | 0.430 |
| corr-cluster weights | 0.502 | 0.406 |

**Conclusion:** bond/diversifier packs and corr-cluster weighting collapse once
financing is applied. Equity packs (`--semis`, `--qqq`, `--network-momentum`) are
robust because they do not rely on hidden leverage.

---

## 7. Six research/patent optimizations

Implemented and evaluated under 3× cap + 1% financing.

| Configuration | Net SR | WF OOS SR | Max DD | Verdict |
|---|---:|---:|---:|---|
| Baseline | 0.539 | 0.427 | −33.2% | — |
| + calibration smooth | 0.539 | 0.427 | −33.2% | Neutral |
| + drawdown control | 0.510 | 0.417 | −26.2% | Risk-only improvement |
| + trend-strength filter | 0.536 | 0.428 | −31.4% | Neutral |
| + network momentum | 0.543 | 0.422 | −35.0% | Already validated |
| + drawdown + trend | 0.496 | 0.444 | −26.2% | No OOS improvement |
| + all four levers | 0.505 | 0.435 | −26.2% | No OOS improvement |

**Diagnostic features also shipped:**
- Drift decomposition (α / β-gap / residual) in `monitor.decompose_drift`
- Worst-quartile edge-decay flag (`--alarm-on-worst-quartile`)
- Warm-up / stateful-restart parity guard

**Decision:** none improved walk-forward OOS Sharpe, so all remain **opt-in /
diagnostic only**. No full Deflated-Sharpe / block-bootstrap / placebo bar was run
because the first OOS check already failed.

---

## 8. Honest validation battery

### Phase 0 edge gate (2007–2026): ✅ PASS — razor-thin

| Gate | Result |
|---|---|
| H1 clears_noise | net 0.71 vs noise-floor 0.40 ✅ |
| H2 edge_real | bootstrap P5 0.38 > 0 ✅ |
| H3 passes_deflated | net 0.71 vs deflated-max 0.69 ✅ (clears by 0.02) |
| R1 cpcv_robust | OOS P5 0.48, 0% paths<0 ✅ |
| R2 walk_forward_ok | OOS 0.62, gap 0.16 ✅ |
| R3 cost_headroom | positive past 25 bps ✅ |

### Long-history 1999–2026: stronger per-config, but config-search overfitting warning

| Era | Sharpe | Vol | MaxDD |
|---|---:|---:|---:|
| Full 1999–2026 | **0.74** | 20.6% | −38% |
| 2007–26 full basket | 0.74 | 21.5% | −38% |
| GFC 2007–09 | 0.88 | 21.3% | −16.8% |
| 2010s | 0.68 | 21.3% | −26.4% |
| COVID 2020 | 1.93 | 22.4% | −14.9% |
| 2021–22 | 0.86 | 23.1% | −18.2% |
| **2023–26** | **0.34** | 21.0% | −30.6% |

- H3 clears more comfortably with more data (N=6921).
- PBO across 14 configs = **0.80** ⚠️ — the IS-best config is below OOS median more
  often than chance, warning config-search overfitting.
- Walk-forward gap near zero, CPCV robust.

**Verdict:** good enough to keep as the live candidate; forward data on real
futures is the only unfakeable test.

---

## 9. Forward deployment status

| Component | Status |
|---|---|
| Daily target generator (`scripts/generate_targets.py`) | ✅ |
| No-broker shadow book (`scripts/shadow_book.py`) | ✅ |
| Daily reconciliation (`scripts/reconcile.py`) | ✅ |
| Kill switch (`data/kill_switch.json`) | ✅ |
| launchd scheduler (`scripts/forward_loop.sh`) | ✅ |
| Optional Alpaca paper executor (`scripts/execute_alpaca.py`) | ✅ |
| Point-in-time feature store + lineage (Phase 2) | ✅ |
| Replay-based decision drift detection (Phase 3) | ✅ |

Current stage: **shadow book only** (`execute_alpaca.py` commented out in
`forward_loop.sh`) until live returns confirm tracking.

---

## 10. Key data artifacts

| File | Contents |
|---|---|
| `data/prices_*.parquet` | Price panels (universe, expanded, long, options_experiment) |
| `data/cot_signal_*.parquet` | COT signal panels |
| `data/experiments.jsonl` | Logged experiment configs (70+ unique configs) |
| `data/options_evaluation_financing.json` | Options-pack evaluation under financing |
| `data/options_evaluation_sensitivity_grid.json` | Cap × financing grid |
| `data/options_evaluation_optimizations.json` | Six-optimization evaluation |
| `data/live_targets.jsonl` | Daily target records |
| `data/live_returns.csv` | Shadow/live daily returns |
| `data/reconciliation/YYYY-MM-DD.json` | Daily reconciliation reports |
| `data/kill_switch.json` | Pause flag |
| `data/broker_orders.jsonl` | Submitted Alpaca orders |
| `data/feature_snapshots/` | Point-in-time feature snapshots |

---

## 11. Bottom-line verdict

- **There is a real ~0.6 Sharpe edge** on ETF proxies, surviving placebo, Lo CI,
  deflated Sharpe, block bootstrap, and walk-forward OOS.
- **The default is robust:** core 19 + governor + 30% buffer.
- **Free signals that cleared the walk-forward:** COT, network momentum, semis,
  QQQ.
- **Financing is the dominant real-world constraint:** many "free Sharpe"
  additions are actually levered bond trades that collapse once a 1% spread is
  charged.
- **Six recent optimizations are diagnostics, not upgrades:** none improved OOS
  Sharpe under an honest financed bar.
- **The honest remaining test is forward data on real futures.** Everything else
  is in-sample or proxy-based.
