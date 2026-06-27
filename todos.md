# signal-engine — next best items

**Status (2026-06-27):** real-data (2007–2026, 19 ETF proxies) net Sharpe **0.65**,
MaxDD −36%, vol on target (21%), clears placebo + Lo CI. Validated win: the
realised-vol **governor**. Confirmed dead-end: **asset-class cluster weighting**.

Rule for everything below: **each new lever must beat the random-walk placebo and
survive an honest OOS split before it ships.** That discipline — not cleverness —
is the whole reason this engine beats the 0.15-Sharpe project it replaced.

---

## ✅ Done
- [x] Core trend engine: EWMAC ×3 + breakout, vol-target sizing, FDM + IDM.
- [x] Rigor suite: Lo CI, Deflated Sharpe, block-bootstrap MC, random-walk placebo.
- [x] Real-data path (yfinance → parquet cache) + 70/30 chronological OOS.
- [x] **Realised-vol governor** — ablation win: Sharpe 0.54→0.65, MaxDD −49%→−36%,
      Calmar 0.23→0.34, vol 25%→21%. On by default.

## 🗑️ Confirmed dead-end (kept as a research flag, OFF by default)
- **Asset-class cluster weighting** (`--cluster-weights`): hurt across the board
  (Sharpe 0.54→0.49, OOS 0.57→0.42) — equal-per-cluster overweights singleton/small
  clusters holding weak names (the lone −0.11-Sharpe REIT, the 2-name credit sleeve).
  The *idea* (don't equal-weight correlated bets) is sound; the asset-class
  *implementation* is naive. → see Tier 1 #2.

---

## Tier 1 — highest expected value (do next)

1. [ ] **Real carry via futures term structure.** The "trend/**carry**" thesis is
   currently trend-only because free ETF data has no term structure. Carry is the
   orthogonal, low-correlation return stream that should lift the diversification
   ratio (2.2× today). *Needs:* a paid futures feed (CSI / Norgate / IBKR) for
   front+deferred contracts. `rules.carry_forecast` + `Instrument.carry_kind`
   already anticipate it. **Biggest genuine-edge lever in the project.**

2. [ ] **Correlation/Sharpe-aware instrument weighting.** Replace equal-weight with
   weights derived from the realised correlation matrix (risk-parity / Carver
   handcrafting by correlation clusters), and/or down-weight chronically
   negative-Sharpe instruments. This recovers the cluster-weighting *upside*
   without the singleton-overweight bug that killed the asset-class version.
   *Validate against the same ablation table.*

3. [ ] **Walk-forward / purged CV instead of one 70/30 split.** The current OOS is a
   single chronological cut; the parent project's hardest-won lesson is that a
   single split sells false dawns (and our IS/OOS gap is a wide +0.20). Port the
   parent's purged expanding-window walk-forward (embargo) to characterise whether
   that gap is regime or fragility. *Highest-value honesty upgrade.*

## Tier 2 — refinements (cheap; validate each vs placebo)

4. [ ] **Governor smoothing.** The governor added turnover (47x→61x) by trading the
   leverage multiplier daily. Smooth it (short EWMA or a buffer band on the
   multiplier) to cut turnover with minimal Sharpe loss. Likely a quick net win.

5. [ ] **Per-instrument cost model.** Flat 1.5 bps today. Real spreads differ
   (USO/EM ETFs ≫ SPY); essential before the futures upgrade. Drives the honest
   net curve.

6. [ ] **Additional orthogonal rules.** Acceleration, longer breakout channels,
   cross-sectional (relative) momentum across the universe. Each ORTHOGONAL rule
   adds free Sharpe via FDM — but only ship the ones that clear the placebo at the
   target horizon (short-horizon rules decayed and *hurt* in the parent's §86 work).

## Tier 3 — honesty / infrastructure

7. [ ] **Real trial counter for Deflated Sharpe.** `n_trials` is a hardcoded 100
   placeholder. Log every config run (like the parent's `_count_experiments`) so the
   Deflated bar reflects the *actual* search as the project grows. Keeps us from
   becoming the thing we replaced.

8. [ ] **Correlation-spike de-risking overlay.** Diversification fails exactly when
   you need it (crises → correlations →1). Add a portfolio-level de-gross when
   average pairwise correlation spikes. Defends the −36% drawdown.

## Tier 4 — deferred (needs paid data) / comes LAST

9. [ ] **Expand to 40–60 real futures markets.** The diversification edge scales
   directly with the number of uncorrelated bets; 19 ETFs is thin. Real futures add
   grains, softs, energy complex, global rates & metals. *Needs paid PIT feed
   (ties to #1).*

10. [ ] **Live execution layer — LAST, not first.** Broker integration, position
    reconciliation, and a tight live-vs-backtest agreement harness. Only after
    Tier 1–3 validate. Writing this before the edge is proven is the original
    project's mistake; do not repeat it.
