# Reusable research levers (2024–2026 sweep)

> Filter applied: only techniques the engine does **not** already have, ranked by
> (value × fit-to-thesis × low overfit-risk). Each maps to a file/function.
> Overfit-risk is called out in your own honesty idiom — a lever that can't clear
> the walk-forward + placebo is not a lever.

## Already covered (do NOT re-add)
Multi-speed EWMAC ensemble, breakout, acceleration, cross-sectional momentum,
carry (`rules.py`); FDM (`forecast.py`); handcrafting by asset-class **and**
correlation cluster, plus Sharpe-adjusted weights (`weights.py`); IDM, vol
targeting, realised-vol governor, no-trade buffer; purged/embargoed walk-forward,
placebo, deflated Sharpe, Lo CI, block bootstrap (`validation.py`); regime/credit/VIX
overlays; COT. The sweep confirms these are the right primitives — the gaps below
are the additive part.

---

## TIER 1 — build these (offline, honesty-enhancing, directly serve the thesis)

### 1. Combinatorial Purged Cross-Validation (CPCV) + Probability of Backtest Overfitting (PBO)
Your `purged_walk_forward` produces **one** OOS path (5 expanding splits). CPCV
(López de Prado; a 2024 ScienceDirect study finds it strictly dominates k-fold and
edges walk-forward on overfit mitigation) partitions the timeline into N groups and
tests over **all C(N,k) train/test combinations**, yielding a *distribution* of OOS
Sharpe instead of a single number — and lets you compute **PBO**: the fraction of
combinations in which the in-sample-best config lands below the OOS median. PBO is
the direct, quantitative answer to "did we overfit the config search?"
- **Why it matters most:** the parent engine died by failing Deflated Sharpe at the
  honest trial count. PBO is the metric that would have caught it *before* deploy.
- **Where:** new `combinatorial_purged_cv()` in `validation.py`, reusing the existing
  purge/embargo logic. Report PBO alongside the deflated max in `experiment_results.md`.
- **Overfit risk:** none — it's a validation tool, not a signal. Pure upside.

### 2. Effective Number of Bets (ENB) — audit the core thesis
Your one-sentence edge is "~20 *uncorrelated* streams." The 2025 "Diversification
Hides Redundancy" paper (arXiv 2510.23150) shows 50 trend instruments often collapse
to **3–4 independent sources** — apparent diversification masking redundancy. You
report IDM 2.13 and div-ratio 2.4× but never measure the **effective number of
independent bets** (Meucci: entropy of the eigenvalue distribution of the return/
forecast correlation matrix, ENB = exp(−Σ pᵢ ln pᵢ)).
- **Why it matters:** if ENB ≈ 4 while IDM assumes far more, your risk target is
  overstated and a "diversifying" instrument may be dead weight. This is a
  thesis-level honesty check, not a tweak.
- **Where:** new `effective_bets()` in `diagnostics.py`; print ENB next to IDM in the
  report. Optionally gate new-instrument admission on *marginal* ENB contribution
  (the paper's rolling redundancy-detection recommendation) rather than asset class.
- **Overfit risk:** none — diagnostic. May *reduce* your universe, which is honest.

---

## TIER 2 — test as opt-in levers (real edge potential, must clear walk-forward)

### 3. Network momentum (a new, price-only signal family)
"Follow the Leader" (arXiv 2501.07135): build a lead-lag graph across instruments
from **price data only** (signature Lévy-area or DTW), sparsify to a directed
adjacency matrix, then each instrument's network-momentum signal is the adjacency-
weighted sum of its leaders' time-series momentum. Reported +26% Sharpe (synthetic)
and 0.35 vs 0.23 (real, 2005–2024) over a MACD baseline.
- **Why it fits:** it's a genuinely *new, weakly-correlated* return stream from data
  you already hold — exactly the "stack uncorrelated bets" thesis. Not an overlay on
  an existing rule; a new forecast in the combine step.
- **Where:** new `network_momentum_forecast()` in `rules.py`, added to the forecast
  dict so FDM handles it; opt-in flag like COT/VIX. Start with the simplest DTW lag
  detector before the signature method.
- **Overfit risk:** medium — the graph-learning step has knobs. Run it through
  `placebo_sharpes` and CPCV before believing it; keep opt-in until forward evidence.

### 4. Regime overlay lookahead audit + conditional transitions
The regime literature's flagged failure mode is using **smoothed** HMM probabilities
(which peek at future data) — an instant backtest inflation. Two actions:
- **Audit** `macro.py`'s regime overlay to confirm it uses **filtered/causal**
  state probabilities only, never smoothed. If it uses smoothed, that alone could
  explain the overlay's OOS attractiveness (`regime_overlay` gap +0.19 in your table).
- **Upgrade:** time-varying transition probabilities conditioned on observable macro
  covariates (credit spreads, vol) make the regime call more responsive without
  lookahead.
- **Overfit risk:** the audit *removes* risk; the conditional-transition upgrade adds
  knobs — gate it.

---

## TIER 3 — marginal / handle with care

- **Drawdown-state control:** move-to-cash / de-risk on a realised drawdown threshold,
  re-risk at recovery (2024 finding: +9.8% avg in the 12m after >10% DDs). You already
  have the vol governor doing continuous de-risking; a discrete DD state is additive
  but easy to overfit the threshold. **Implemented** as `portfolio.drawdown_overlay`
  and `Config.use_drawdown_control`; tested opt-in only. Lowers MaxDD but leaves
  risk-adjusted return roughly unchanged.
- **Weight persistence across recalibration:** enforce gradual weight changes across
  the expanding-window refits to cut estimation-driven turnover (research: improves
  stability). Cheap, low-risk, small effect. **Implemented** as
  `Config.calibration_smooth` and `_smooth_parameter_transitions` in `backtest.py`;
  neutral on aggregate performance in this dataset.
- **Trend-strength filter:** de-gear when the average absolute combined forecast is in
  the weakest historical percentile, targeting low-trend regimes such as 2023–26.
  **Implemented** as `portfolio.trend_strength_overlay` and
  `Config.use_trend_strength_filter`; neutral on the full sample and does not recover
  the recent weakness with default parameters.
- **Volatility parity vs vol targeting** for position sizing: an alternative to your
  vol-target sizing; the 2024 evidence is that vol-targeting alpha ≈ a trend loading,
  so don't expect free Sharpe — likely inert on your walk-forward. Low priority.

---

## Implementation status (2026-07-15 — all shipped)

| Lever | Where | How to use | Status |
|---|---|---|---|
| CPCV | `validation.combinatorial_purged_cv(prices, cfg, n_groups, k_test)` | returns OOS-Sharpe distribution + `pct_paths_below_zero` | shipped |
| PBO | `validation.probability_backtest_overfitting(returns_matrix)` | pass a T×N panel of candidate-config returns → `pbo` | shipped |
| Honest n_trials | `validation.register_trial(cfg)` / `honest_n_trials()` | unions `data/trial_registry.jsonl` and `data/experiments.jsonl`, deduplicates by effective Config | shipped, corrected 2026-07-15 |
| Lookahead guard | `validation.assert_no_lookahead(fn, data)` | raises if history changes when the tail is revealed | shipped |
| Effective bets | `diagnostics.effective_number_of_bets(returns)` / `diversification_audit(result)` | ENB + `idm_vs_effective` | shipped |
| Network momentum | `Config(use_network_momentum=True)` / `--network-momentum` | new price-only rule wired through FDM | **VALIDATED-POSITIVE** |
| Drift decomposition | `monitor.decompose_drift` (auto-included in `reconcile`) | α / β-gap / residual split | shipped |
| Quartile edge-decay | `monitor.edge_decay_report` → `worst_quartile`, `decay_warning`; `--alarm-on-worst-quartile` | kill-switch input + diagnostic | shipped |
| Champion/challenger | `generate_targets.py --challenger`; `live.champion_challenger_report()` | parallel books, promote on forward evidence | shipped |
| Arrival slippage | `live.compute_delay_slippage`; `append_shadow_return(arrival_prices=...)` | records `delay_return` | shipped |
| Stabilised gross cap | `execute_alpaca.py --equity-buffer --equity-ref-halflife` | buffer + smoothed-equity anchor | shipped |
| Calibration smoothing | `Config.calibration_smooth` / `--calibration-smooth` | gradual weight/IDM/FDM transitions | shipped, neutral |
| Drawdown control | `Config.use_drawdown_control` / `--drawdown-control` | causal DD-state de-grossing | shipped, risk-only |
| Trend-strength filter | `Config.use_trend_strength_filter` / `--trend-strength-filter` | de-gear on weak combined forecasts | shipped, neutral |
| Warm-up / restart parity | `Config.min_history_required()`; guard in `live.generate_target` | errors on insufficient history | shipped |

Regime overlays audited causal (`macro.py`); guard is `validation.assert_no_lookahead`.
Covered by `tests/test_research_levers.py` (26 tests). All tests pass.

### Validation notes on the newer levers

- **Network momentum** was promoted to VALIDATED-POSITIVE after it improved or matched
  walk-forward OOS Sharpe across multiple test configurations; it is included by
  default in the options-evaluation universe.
- **Drawdown control**, **calibration smoothing**, and **trend-strength filter** were
  evaluated under 3× gross cap + 1% financing (`scripts/eval_optimizations.py`). None
  improved OOS Sharpe versus the baseline; they remain **opt-in / diagnostic only**.
- **Quartile edge-decay** and **warm-up parity** are safety/diagnostic features with
  no performance claim.

## Recommended build order
1. **CPCV + PBO** (`validation.py`) — pure honesty upside, catches the failure mode
   that killed the parent engine.
2. **Effective Number of Bets** (`diagnostics.py`) — audits the core diversification
   thesis; may right-size the universe.
3. **Network momentum** (`rules.py`, opt-in) — the one genuinely new *edge* candidate;
   only promote on forward evidence.
4. Regime lookahead audit — cheap correctness check on an existing overlay.

Items 1–2 are offline, deterministic, and directly reinforce the project's discipline;
they're the right first commits.

## Sources
- CPCV / PBO — arXiv/ScienceDirect S0950705124011110 (2024 OOS-method comparison); Bailey & López de Prado, Deflated Sharpe (davidhbailey.com)
- Diversification hides redundancy — arXiv 2510.23150; Meucci "Effective Number of Bets"
- Network momentum — arXiv 2501.07135 ("Follow the Leader")
- Regime / HMM lookahead — LSEG regime detection; aimspress ensemble-HMM (2025)
- Trend/vol-targeting — Sepp (2025) "Science and Practice of Trend-following"; SSRN 4773781 (vol targeting = trend loading); Concretum (VT vs VP vs pyramiding); Man Group drawdowns
- Portfolio construction — HRP (López de Prado); Carver handcrafting / FDM (qoppac / pysystemtrade)
