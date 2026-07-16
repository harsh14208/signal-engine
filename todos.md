# signal-engine — next best items

**Status (2026-07-15):** real-data (2007–2026, 19 ETF proxies) net Sharpe **0.69**,
MaxDD −38%, vol on target (21.4%); clears placebo + Lo CI, and now **clears Deflated
Sharpe at the honest trial count**. The honest read is the **4-fold walk-forward**
(mean OOS **0.61**, gap +0.12), NOT the single 70/30 split (OOS 0.55).
Validated wins: realised-vol **governor**, **30% position buffer**, **`--cot`**,
**`--semis`**, **`--qqq`**, and **`--network-momentum`**. 🟢 **NEW — financing /
leverage-cost model** (`--financing-rate`, `--financing-threshold`, `--max-gross`)
so bond-pack additions are compared on a like-for-like cost basis. 🟢 **NEW — six
research/patent optimizations implemented as opt-in diagnostics**: drift
decomposition, warm-up parity, quartile edge-decay, calibration smoothing, drawdown
control, and trend-strength filter. **None improved walk-forward OOS Sharpe versus
the financed baseline**, so they remain research flags. Confirmed dead-ends /
opt-in only: **asset-class cluster weighting**, **VIX term-structure overlay**,
**Baa-10Y credit-spread overlay**, **GARCH vol sizing**, **HMM regime overlay**,
**S&P 500 x-sectional momentum sleeve**, **VRP injection (detonated vol-targeting —
parked)**, **drawdown control**, **trend-strength filter**, **calibration smoothing**.
⚠ **`--ship-candidate` is NOT promoted** — single-split false dawn. The validated
default remains **core 19 + governor + 30% buffer**.

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
- [x] **Financing / leverage-cost model** — `Config.financing_rate`,
      `financing_threshold`, `max_annual_financing_cost`; `--max-gross` gross-notional
      cap. A 1% spread reduces Net Sharpe ~0.10–0.13 and makes equity packs (`--semis`,
      `--qqq`, `--network-momentum`) competitive with levered bond packs.
- [x] **Instrument-pack flags** — `--semis`, `--qqq`, `--diversifier-pack`,
      `--rate-pack`. Semis and QQQ are VALIDATED-POSITIVE; bond packs are leverage-
      dependent and must be evaluated with `--financing-rate`.
- [x] **Six research/patent optimizations (diagnostic only)** — drift decomposition,
      warm-up/restart parity, quartile edge-decay alarm, calibration smoothing, drawdown
      control, trend-strength filter. Implemented and evaluated under 3× cap + 1%
      financing; none improved OOS Sharpe versus baseline, so they remain opt-in.

## 🗑️ Confirmed dead-end (kept as a research flag, OFF by default)
- **Asset-class cluster weighting** (`--cluster-weights`): hurt across the board
  (Sharpe 0.54→0.49, OOS 0.57→0.42) — equal-per-cluster overweights singleton/small
  clusters holding weak names (the lone −0.11-Sharpe REIT, the 2-name credit sleeve).
  The *idea* (don't equal-weight correlated bets) is sound; the asset-class
  *implementation* is naive. → see Tier 1 #2.

---

## Tier A — forward-validate the edge (paper deploy + reconcile) — THE next step

The engine is validated in-sample + walk-forward but has seen **NO forward data**. Closing
that gap is what the whole research-harness-first discipline was built for; it's free; and
the reconciliation tooling (`monitor.reconcile` / `monitor.edge_decay_report`) already
exists. Goal: turn "backtested 0.61 WF OOS" into "confirmed forward" and catch live-vs-
backtest divergence early — the exact failure mode the parent project never instrumented.

A1. [x] **Daily target-position generator** (`scripts/generate_targets.py`). Refreshes prices
    + COT, runs the validated config (core 19 + governor + 30% buffer + COT), and appends
    target units/notional/forecast to `data/live_targets.jsonl`. COT refresh falls back to
    cache if the network fails. Added `BacktestResult.buffered` so the live target is the
    post-buffer position to hold next close (no lookahead).

A2. [x] **No-broker shadow paper book first** (`scripts/shadow_book.py`). Marks the next-day
    shadow return using closing prices and appends to `data/live_returns.csv`. No broker
    required — the cheapest way to answer "does forward track the backtest?".

A3. [x] **Daily reconciliation report** (`scripts/reconcile.py`). Compares live vs modeled
    returns via `monitor.reconcile` (corr / tracking error / drift) and
    `monitor.edge_decay_report` (rolling 1y Sharpe). Persists JSON under
    `data/reconciliation/YYYY-MM-DD.json` and prints a markdown summary.

A4. [x] **Schedule it** (launchd). Added `scripts/forward_loop.sh`,
    `scripts/launchd/com.signal.engine.forward.plist`, and `scripts/install_launchd.sh` /
    `uninstall_launchd.sh`. Runs daily at 17:30 local time; idempotent and holiday-safe.

A5. [x] **Optional: Alpaca paper execution** (`scripts/execute_alpaca.py`). Reads latest
    target, computes delta notional vs current Alpaca positions, submits fractional notional
    orders (paper by default; `--live` explicit), and logs fills to
    `data/broker_orders.jsonl`. Respects the kill switch.

A6. [x] **Guardrails / kill-switch.** `scripts/reconcile.py` writes `data/kill_switch.json`
    (`paused: true`) when reconciliation is not aligned or rolling 1y Sharpe drops below
    `--alarm-floor` (default 0.0). `execute_alpaca.py` refuses new orders while paused.

A7. [x] **Forward-confirm `--cot` specifically.** Every target record stores `use_cot`,
    `cot_momentum`, and `cot_as_of`; `live_returns.csv` stores `use_cot`. This lets the
    accruing forward data answer whether the COT walk-forward edge holds.

A8. [x] **Runbook** (`docs/FORWARD.md`): start/stop commands, artifact descriptions, alarm
    meanings, kill-switch handling, and Alpaca credential setup.

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

0j. [~] **Vol-carry / VRP sleeve.** First parked for the WRONG reason (the parent's options
    panel is ~5 days). CORRECTED: VRP needs no options panel — CBOE vol indices ARE
    long-history implied vol (^VIX 1990+, ^RVX/^OVX/^GVZ/^EVZ 2007–08+). Built `vrp_data.py`:
    free synthetic short-vol price per (vol-index → ETF) pair, no-lookahead, tested.
    **The real finding:** injecting a short-vol stream as a *tradable instrument* DETONATES
    the engine's vol-targeting — across 3 constructions (raw / tanh-bounded / ragged-start)
    the book over-levers in calm patches and the equity blows up (MaxDD → −inf, spurious
    Sharpe 4–7; the honesty tooling caught it instantly). VRP is fat-tailed; harvesting it
    needs a DEDICATED, position-capped short-vol sizing path, not instrument injection.
    **Parked** — data module kept + tested, intentionally NOT wired into the CLI.

0k. [x] **Principled regime signal from the parent's HMM + FRED stress indices.** Ported the
    parent's 2-state Gaussian HMM to `macro.py::hmm_regime_overlay()` (features: VIX, SPY
    20d return, 10Y–2Y spread, 30d realised vol). Wired via `--hmm-regime-overlay`.
    **Validation:** inert on the core book (net SR unchanged at 0.69, mean overlay 0.93×).
    Left **opt-in** for regime research; does not crash on degenerate HMM fits.

0l. [x] **Cross-sectional equity-momentum sleeve from the parent's PIT universe.** Built
    `equity_momentum_sleeve.py::build_equity_momentum_sleeve()` using PIT S&P 500 membership
    and `cache_ohlcv/`. Returns a synthetic `SP500_XSMOM` price series, injectable with
    `--equity-momentum-sleeve`. Added outlier clipping after bad ticks blew up the series.
    **Validation:** net SR **0.69 → 0.62**; the sleeve's standalone Sharpe is −0.13 in this
    configuration. Left **opt-in** for research (different deciles/lookbacks may help).

0m. [x] **Real bond carry + a curve trade from the full FRED Treasury curve.** Upgraded
    `carry_data.py` with tenor-specific roll-down proxies from FRED `DGS2/5/10/30`
    (`--real-bond-carry`). Added `curve_data.py::load_2s10s_steepener()` and a synthetic
    `UST2S10S` instrument (`--curve-steepener`).
    **Validation:** real bond carry is inert on the core book (net SR 0.69). The 2s10s
    steepener improves the single split (net SR **0.69 → 0.71**, OOS 0.55 → 0.57) but
    doubles turnover to ~93× and the walk-forward is mixed (mean OOS 0.59, gap +0.19,
    last fold 0.25). Left **opt-in** pending cost-aware refinement.

0n. [x] **GARCH(1,1) forward-vol sizing.** Added optional `arch` GARCH(1,1)-t blend in
    `volatility.py`, wired through `--garch-vol` / `--garch-weight`. Requires the `[garch]
    optional dependency.
    **Validation:** a 50% GARCH blend hurts OOS (net SR 0.67 vs 0.69, OOS 0.49 vs 0.55,
    gap +0.26). Left **opt-in research flag only**.

0o. [x] **No-broker monitoring harness (`monitor.py`, `--monitor`).** The cheap half of the
    live-vs-backtest reconciliation the parent never had: rolling 1-year Sharpe with an
    edge-decay alarm, plus `reconcile(live, backtest)` (correlation / tracking error / drift)
    ready for the day live returns exist. The default's rolling-Sharpe path is healthy.

0p. [x] **Flag taxonomy (optionality cleanup).** `--help` now ends with a taxonomy splitting
    flags into CORE (validated) / RESEARCH (tested, none beat the walk-forward default) /
    VALIDATION-DIAGNOSTICS, so the default path stays obvious despite ~70 flags. Deeper
    pruning is deferred (project philosophy keeps tested dead-ends as opt-in research flags).

0q. [x] 🟢 **COT positioning (`cot_data.py`, `--cot`) — the FIRST free lever to clear the
    walk-forward.** Free CFTC Commitments-of-Traders (Socrata `6dca-aqww`, 1986+): weekly
    net non-commercial / open interest, mapped to ~10 macro-core ETFs (S&P, Russell, gold,
    silver, crude, bonds, notes, EUR, JPY, USD) by max-OI contract per date. Turned into a
    PRE-SPECIFIED contrarian forecast (fade crowded specs = side with commercials) — a
    per-instrument RULE combined via FDM (not a de-gross overlay). z-score on full history
    so it survives the walk-forward; threaded through `purged_walk_forward` (the first
    per-instrument feature that is). **Result:** full 0.69→0.72, single-split OOS 0.55→0.60,
    **walk-forward mean OOS 0.61→0.63**, block-boot P5 0.34→0.38, clears Deflated @29 trials
    + placebo. Modest + fold-concentrated (one 2018–22 fold does much of it) and needs a CFTC
    fetch → kept **opt-in / VALIDATED-POSITIVE**, not auto-promoted to default.
    **Broadening coverage via `--expanded-universe --cot` tested → INERT on the walk-forward
    (mean OOS 0.54→0.54, full 0.70→0.69):** the COT-mapped instruments fall from ~53% of the
    core book to ~33% of the expanded book, so the positioning signal dilutes, and the
    expanded base is itself weaker. COT's value is **concentrated on the core macro universe**,
    not broadened. **Surgical "core + UNG/CORN/WEAT" (`--core-commodities`) tested too → also
    fails:** the commodities HURT the base (core 0.61 → 0.59 WF OOS — natgas decay + young
    CORN/WEAT histories, the same IS-flattering trap), and COT on the bigger set is inert
    (0.59→0.59). Best config stays **core 19 + COT (0.63 WF OOS)**; `--core-commodities` kept
    as a RESEARCH-inert flag. **Sign A/B done (`--cot-momentum`): on the walk-forward contrarian and
    momentum are INDISTINGUISHABLE (both 0.63 mean OOS, +0.15 gap); contrarian is marginally better
    only on the lower-power full-sample (0.72 vs 0.70, Calmar 0.38 vs 0.34) — consistent with the
    pre-specified rationale, so the contrarian default stands (flipping on a backtest = sign-shop).**
    Only remaining COT lever: forward-confirm the live edge via `--monitor` as data accrues.

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
