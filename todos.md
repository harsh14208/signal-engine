# signal-engine — next best items

**Status (2026-06-27):** real-data (2007–2026, 19 ETF proxies) net Sharpe **0.68**,
MaxDD −36%, vol on target (21%); clears placebo + Lo CI, and now **clears Deflated
Sharpe at the honest 16-trial count (0.68 > 0.54)**. The honest read is the **4-fold
walk-forward** (mean OOS **0.59**, gap +0.14), NOT the single 70/30 split (OOS 0.51).
Validated wins: realised-vol **governor** and **30% position buffer**. Confirmed
dead-ends: **asset-class cluster weighting**, **VIX term-structure overlay**, **Baa-10Y
credit-spread overlay**. ⚠ **`--ship-candidate` (expanded universe + regime overlay) is
NOT promoted** — it looks great on the single split (OOS 0.72 / gap +0.03) but the
walk-forward refutes it (mean OOS **0.54 < the default's 0.59**, gap +0.28); a
single-split false dawn. The validated config is the **default (core 19 + governor)**.

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
- [x] **Honesty bug sweep** — default `run_backtest` now uses expanding-window
      calibration for weights/IDM/FDM; `.bfill()` lookahead removed from buffer and
      governor; first-trade cost is now charged; `source="cache"` raises on missing
      symbols; single-symbol yfinance column renamed correctly; expanded-universe
      metadata wired into weights, reports, and carry assignment.
- [x] **Default buffer raised to 30%** — a pure parameter change that raised baseline
      OOS Sharpe from 0.51 to 0.55 and cut turnover from ~60× to ~47×.
- [x] **`--ship-candidate` preset (implemented; NOT promoted)** — expanded universe +
      regime overlay + 30% buffer + regime smooth 5. Looks like Net SR 0.74 / OOS 0.72 /
      gap +0.03 on a single 70/30 split, but the **4-fold walk-forward refutes it**
      (mean OOS 0.54 < the plain default's 0.59, gap +0.28). A single-split false dawn —
      kept as a research flag only. Lesson: promote on the walk-forward, never one split.

## 🗑️ Confirmed dead-end (kept as a research flag, OFF by default)
- **Asset-class cluster weighting** (`--cluster-weights`): hurt across the board
  (Sharpe 0.54→0.49, OOS 0.57→0.42) — equal-per-cluster overweights singleton/small
  clusters holding weak names (the lone −0.11-Sharpe REIT, the 2-name credit sleeve).
  The *idea* (don't equal-weight correlated bets) is sound; the asset-class
  *implementation* is naive. → see Tier 1 #2.

---

## Tier 0 — buildable NOW on data already in hand (no paid feed)

Need only the cached 19-ETF panel, free yfinance, and the parent project's existing
free macro data (FRED, VIX, the 2-state regime HMM). **Several of these outrank the
paid-data items below — do them first.** Same rule applies: clear the placebo and an
honest OOS split before shipping.

0a. [x] **Free carry proxies → switch the dormant carry rule ON.** Implemented in
    `carry_data.py`: FRED `T10Y3M` for bond carry, yfinance trailing 12-month dividend yield
    for equity/real-estate/credit carry, and a `DGS3MO` stub for FX (foreign FRED coverage is
    spotty). Enabled with `--carry-proxies`.
    **Validation (cache, 2007–2026):** net Sharpe 0.65→0.66, OOS 0.51→0.51, placebo clears.
    Positive but marginal; left **opt-in** so the offline synthetic demo stays network-free.

0b. [x] **Expand the universe 19 → ~40 free ETFs.** Added `EXPANDED_UNIVERSE` in
    `markets.py` (~42 ETFs) and `--expanded-universe` flag. The >300-bar filter drops thin
    names automatically; a separate `prices_expanded.parquet` cache keeps the core cache clean.
    **Validation (yfinance/cache, 2007–2026):** net Sharpe 0.65→0.70, OOS 0.51→0.52, placebo
    clears, but turnover rose 61x→79x and the IS/OOS gap widened +0.20→+0.27. Left **opt-in**
    pending a cost-aware re-balancing of the expanded set.

0c. [x] **Empirical expanding-window forecast scalars.** Implemented in `scalars.py` and
    wired via `--empirical-scalars`. Each rule's forecast is rescaled by
    `target / expanding_mean|forecast|.shift(1)` so the vol overshoot is fixed at the source.
    **Validation (cache, 2007–2026):** net Sharpe flat 0.65, OOS 0.51→0.49, gap +0.20→+0.24.
    Does not survive the honest OOS split; left **opt-in** for further research.

0d. [x] **Macro-regime overlay from free data.** Implemented in `macro.py`: VIX from yfinance
    (`^VIX`), NFCI from FRED, and a `regime_overlay` that de-gears when VIX spikes or the
    equal-weighted equity index is in drawdown. Enabled with `--regime-overlay`.
    **Validation (cache, 2007–2026):** net Sharpe flat 0.65, OOS 0.51→0.53, gap +0.20→+0.18,
    placebo clears, but turnover jumped 61x→69x and MaxDD was not materially improved. Left
    **opt-in** because the turnover cost offset the OOS gain.

0e. [x] **Cross-sectional (relative) momentum rule.** Implemented as `rules.cross_sectional_momentum_forecast`;
    enabled with `--xsmom`. It ranks recent total returns across the panel and maps rank to
    `[-20, 20]`. Added via FDM; placebo-clear on synthetic data.

0f. [x] **Correlation-clustered / risk-parity weights.** Implemented as `weights.corr_cluster_weights`
    and `weights.sharpe_adjusted_weights`, selectable with `--weight-scheme corr_cluster|sharpe`.
    Uses correlation-threshold union-find clustering to avoid the singleton-overweight bug that
    killed asset-class clustering.

0g. [x] **VIX term-structure overlay.** Implemented in `macro.py`: free yfinance data
    for `^VIX`, `^VIX9D`, `^VIX3M`; `vix_term_overlay` gears up when the short end is
    calm (`vix3m/vix` low) and de-gears when near-term fear spikes (`vix9d/vix` high).
    Enabled with `--vix-term-overlay` (and `--vix-term-smooth`).
    **Validation (cache, 2007–2026, added to ship candidate):** net SR 0.74→0.72,
    OOS 0.72→0.68, gap +0.03→+0.06, turnover 62.7x→63.8x. Does not beat the ship
    candidate; left **opt-in** for further research.

0h. [x] **Credit-spread overlay.** Implemented in `macro.py`: Moody's Baa corporate
    yield minus 10-year Treasury (`BAA10Y`) from FRED as a long-history, free credit
    risk premium; `credit_overlay` gears up/de-gears based on the spread's ratio to
    its trailing median. Enabled with `--credit-overlay` (and `--credit-smooth`).
    **Validation (cache, 2007–2026, added to ship candidate):** with a small grid
    search the best tuned setup (upper=1.3, lower=0.7, lookback=756) is statistically
    tied with ship alone (net SR 0.74, OOS 0.72, gap +0.04, turnover 62.9x). Because
    the threshold was fit on the same sample, the tie is treated as **no improvement**.
    Left **opt-in** for further research.

0i. [x] **Diagnostics on the data we already have.** Implemented in `diagnostics.py` and
    exposed via `--diagnostics`:
    • Cost × buffer frontier (net Sharpe, IS/OOS Sharpe, turnover, MaxDD across a grid).
    • Per-instrument gross/cost/net attribution.
    • VIX-regime split (high vs low VIX Sharpe/vol/MaxDD).
    **Result:** the +0.20 IS/OOS gap is clearly regime-driven — low-VIX periods have Sharpe
    ~1.28, high-VIX periods ~0.08. This is the same stress-regime story the overlay tries to
    address; the diagnostics confirm the source of the gap.

## Tier 0b — leverage the *parent project's* existing data (free, no new feed)

The sibling `TradingRecommendationSystem` repo already has years of free/cached data we
can borrow without any paid feed. Each is a research lever — **validate on the
walk-forward (not a single split) before promoting** (see the ship-candidate lesson above).

0j. [ ] **Vol-carry / VRP sleeve from the parent's options-IV data.** The parent has
    `data/cache_options/options_iv_history.parquet`, an ORATS opportunity model, and a
    demonstrated VRP edge (implied − realised vol). On the index ETFs we already trade
    (SPY/EFA/EEM/…), a short-vol-when-IV≫RV forecast is a genuine, low-correlation,
    *carry-like* return stream — the closest free substitute for the paid-futures carry in
    Tier 1 #1. **Highest-value item here:** real orthogonal alpha, data already on disk.
    Model P&L realistically (option cost/assignment); deploy as a forecast sleeve, not a
    separate options book.

0k. [ ] **Principled regime signal from the parent's HMM + FRED stress indices.** Our
    current regime overlay (a VIX threshold) is ~inert on the walk-forward. The parent
    already computes a 2-state `hmmlearn` macro-regime HMM on SPY+VIX (`macro_regime.py`)
    and pulls NFCI + STLFSI4 from FRED. Port the HMM bull-probability / NFCI as the
    de-gross signal and re-test — a better regime input may make the overlay finally earn
    its place (validate on walk-forward; the VIX-threshold version did not).

0l. [ ] **Cross-sectional equity-momentum sleeve from the parent's PIT universe.** The
    parent has survivorship-corrected S&P 500 point-in-time membership
    (`data/sp500_ticker_start_end.csv`), a deep `cache_ohlcv/` history, and a built
    cross-sectional harness (`cross_sectional_alpha_model.py`). A dollar-neutral
    top/bottom-decile equity-momentum sleeve is orthogonal to the macro-asset trend book
    and stacks via IDM — all on free data already cached next door. Size it as ONE cluster
    (internal equity correlation is high).

0m. [ ] **Real bond carry + a curve trade from the full FRED Treasury curve.** Upgrade the
    bond-carry proxy (0a) from a slope sign to actual roll-down using DGS2/5/10/30 (free
    FRED; the parent already pulls T10Y3M), and add a 2s10s curve steepener as a synthetic
    instrument — a low-correlation macro bet with decades of free history.

0n. [ ] **GARCH(1,1) forward-vol sizing.** The parent already wired `arch` for conditional-
    vol forecasts. A forward-vol estimate could sharpen the governor / position sizing vs
    the trailing EW vol. Cheap to A/B — but only keep it if it beats EW on the walk-forward.

## Tier 1 — highest expected value (do next)

1. [ ] **Real carry via futures term structure.** The "trend/**carry**" thesis is
   currently trend-only because free ETF data has no term structure. Carry is the
   orthogonal, low-correlation return stream that should lift the diversification
   ratio (2.2× today). *Needs:* a paid futures feed (CSI / Norgate / IBKR) for
   front+deferred contracts. `rules.carry_forecast` + `Instrument.carry_kind`
   already anticipate it. **Biggest genuine-edge lever in the project.**

2. [x] **Correlation/Sharpe-aware instrument weighting.** See Tier 0 #0f above. Available
   via `--weight-scheme corr_cluster` and `--weight-scheme sharpe`; default remains equal-weight.

3. [x] **Walk-forward / purged CV instead of one 70/30 split.** Implemented in
   `validation.purged_walk_forward`; exposed as `--walk-forward N` in the CLI. Uses expanding
   training windows with an embargo gap; OOS fold uses weights/IDM/FDM estimated only on the
   preceding train window.

## Tier 2 — refinements (cheap; validate each vs placebo)

4. [x] **Governor smoothing.** Implemented in `portfolio.vol_governor` via the `smooth` argument
   and `--governor-smooth SPAN` CLI flag. EWMA is applied to the already-lagged multiplier, so
   there is no lookahead.

5. [x] **Per-instrument cost model.** Added `Instrument.cost_bps` in `markets.py` with plausible
   ETF spreads. Use `--cost-scheme instrument` to apply them; default `--cost-scheme flat` keeps
   the previous 1.5 bps behaviour.

6. [~] **Additional orthogonal rules.** Acceleration (`--accel`) and cross-sectional momentum
   (`--xsmom`) are implemented. Longer breakout channels remain available via the existing
   configurable `breakout_spans`. Validate each rule vs placebo before enabling by default.

## Tier 3 — honesty / infrastructure

7. [x] **Real trial counter for Deflated Sharpe.** Added `experiments.py`: every run is
   appended to `data/experiments.jsonl` with a config hash, and `--n-trials` now defaults to
   the number of unique configs searched (+1 for the current run). The old hardcoded 100 is
   still available by passing `--n-trials 100`.

8. [x] **Correlation-spike de-risking overlay.** Implemented in `portfolio.corr_spike_overlay`;
   enabled with `--corr-spike`. Computes a lagged rolling average pairwise correlation and
   de-grosses the book when it spikes above a threshold.

## Tier 4 — deferred (needs paid data) / comes LAST

9. [ ] **Expand to 40–60 real futures markets.** The diversification edge scales
   directly with the number of uncorrelated bets; 19 ETFs is thin. Real futures add
   grains, softs, energy complex, global rates & metals. *Needs paid PIT feed
   (ties to #1).*

10. [ ] **Live execution layer — LAST, not first.** Broker integration, position
    reconciliation, and a tight live-vs-backtest agreement harness. Only after
    Tier 1–3 validate. Writing this before the edge is proven is the original
    project's mistake; do not repeat it.
