# signal-engine

A diversified systematic **futures trend + carry** engine (Carver / AHL style) —
now with a full forward-testing / paper-trading deployment on top of the
research core.

> **The one-sentence edge:** many small, *uncorrelated* risk-adjusted bets stack
> into a portfolio Sharpe far higher than any single bet — because
> diversification is the only free lunch in markets.

This is deliberately the opposite of a single, heavily-tuned equity strategy. No
instrument here is expected to be impressive on its own (~0.2–0.4 standalone
Sharpe). The edge is in combining ~20 weakly-correlated return streams at a
constant risk target — and that combination is *mathematics* (the IDM below),
not a forecast you can over-fit.

---

## Why this project exists (the lesson it's built on)

It is a clean-room rebuild that fixes the five structural mistakes of a previous
large-cap-equity mean-reversion engine (~18,000 lines, 96 research sections, 10
sizing layers) whose own honest statistics topped out at a forward Sharpe of
**~0.15** and **failed its own Deflated-Sharpe test** at the real trial count:

1. **Wrong market.** Large-cap US equity mean-reversion on free OHLCV is the most
   strip-mined corner of markets — near-zero prior probability of durable retail
   edge. → *Here:* trend + carry across uncorrelated asset classes, where the
   edge is structural (risk premia + behavioural under/over-reaction) and
   diversification does the heavy lifting.
2. **Optimised the search, not the edge.** 96 gates stacked on a 0.13-OOS
   strategy = overfitting by construction. → *Here:* a handful of rules, fixed
   published scalars (no in-sample fitting), and a **pre-registered OOS / placebo
   protocol you run before believing anything** (`validation.py`).
3. **Spent effort on entry; the edge was in cost/holding structure.** → *Here:*
   turnover is a first-class control (no-trade **buffer**, monthly-ish trend
   speeds) and cost robustness is a headline metric.
4. **Built on broken free data and never reconciled live vs backtest.** → *Here:*
   a tiny, honest data layer with a deterministic synthetic generator, a
   point-in-time price/COT cache that never silently rewrites history, and —
   as of the forward-deployment phase below — a live paper-trading loop that
   reconciles actual returns against the model every day.
5. **No crisp edge thesis (kitchen sink).** → *Here:* one sentence, at the top of
   this file, that every line of code serves.

**Build discipline:** research harness first, execution last, and every lever
promoted on the walk-forward or on forward (live) evidence — never on a single
backtest split. That sequencing is the whole point, and it's now also the whole
point of how new levers get promoted into the default config (see
[Promotion framework](#promotion-framework-edge_gatepy) below).

---

## How it works

```
prices ─▶ returns ─▶ blended vol ─▶ rule forecasts ─▶ combine (FDM)
       ─▶ vol-target sizing (IDM) ─▶ overlays (regime/corr/COT) ─▶ governor
       ─▶ no-trade buffer ─▶ shift(1) ─▶ P&L − costs − financing
```

| Concept | Module | What it does |
|---|---|---|
| Volatility | `volatility.py` | Carver blended EW vol (70% recent / 30% long-run), optional GARCH(1,1) blend |
| Trend rules | `rules.py` | EWMAC crossover (3 speeds) + breakout + acceleration + cross-sectional momentum + network (lead-lag) momentum, vol-normalised, scaled to mean \|f\|≈10, capped ±20 |
| Carry rule | `rules.py`, `carry_data.py` | Risk-adjusted annualised carry — synthetic (demo), free proxy (FRED/dividend yield), or real tenor-specific bond roll-down |
| COT rule | `cot_data.py` | Free CFTC Commitments-of-Traders positioning, contrarian sign, PIT-stitched weekly cache — **validated-positive** |
| Combine | `forecast.py` | Weighted sum × **FDM** (Forecast Diversification Multiplier) |
| Sizing | `portfolio.py` | Volatility targeting + **IDM** (Instrument Diversification Multiplier) — *this is where diversification becomes return* |
| Overlays | `portfolio.py`, `macro.py` | Correlation-spike de-risk, regime (VIX/drawdown/HMM), VIX-term structure, credit spread — all opt-in, none currently walk-forward-validated |
| Turnover | `portfolio.py` | No-trade buffer band (expanding, no lookahead) |
| Engine | `backtest.py` | No-lookahead P&L (positions decided at close *t-1*), net of costs and financing |
| Metrics | `metrics.py` | Sharpe, MaxDD, CAGR, Calmar, Sortino, skew, turnover |
| **Rigor** | `validation.py` | Lo (2002) CI · **Deflated Sharpe** (honest trial count) · CPCV + **PBO** · block-bootstrap MC · **random-walk placebo** · paired walk-forward fold comparison |
| **Promotion gate** | `edge_gate.py` | Hard gates (noise/bootstrap/deflated-Sharpe) + robustness gates (CPCV/walk-forward/cost-headroom) + forward-evidence gate — see below |
| Reports | `report.py` | Headline + the **diversification report** (the edge made visible) |

**IDM is the thesis in one line.** A basket of weakly-correlated instruments has
far lower vol than each leg, so the book is scaled up by `IDM = 1/√(w'·Ρ·w)` to
hit the vol target. Low correlation → high IDM → more return per unit of the same
risk. That is "uncorrelated bets stack," stated as arithmetic.

---

## Run it

No install needed beyond `numpy` / `pandas` / `pyarrow`. The default run is
**synthetic and offline**:

```bash
# Offline synthetic demo (deterministic)
python -m signal_engine

# + the full statistical honesty suite (Lo CI, Deflated Sharpe, MC, placebo)
python -m signal_engine --validate

# Chronological in-sample / out-of-sample split, or purged walk-forward
python -m signal_engine --oos 0.7 --validate
python -m signal_engine --walk-forward 5

# Real ETF-proxy data (needs network + `pip install -e .[data]`)
python -m signal_engine --source yfinance --validate
```

Tests:

```bash
pip install -e .[dev]
pytest            # 284 tests across 30 files
ruff check signal_engine tests
```

### What the synthetic demo shows (and does NOT claim)

The synthetic generator (`data.synthetic_prices`) is a **labelled test/demo DGP**
with persistent trends and low correlation — it exists to prove the pipeline and
the diversification *math* are correct and verifiable offline. It is **not** a
claim about live performance — that comes from real data (below). The honesty
tooling is the product; the synthetic number is a fixture.

### Real-data results (`--source cache`/`auto`, 2007–2026, core 19 ETF proxies)

Default config (trend + COT, governor ON, 30% buffer) on ~4,900 days of actual
prices — the live forward loop's own daily reconciliation reports currently show
**modeled full-history Sharpe ≈ 0.71** (gross). The headline backtest table:

| Metric | Value |
|---|---|
| **Net Sharpe** | **~0.69** (gross ~0.71–0.72) |
| Max drawdown | ~ −36% to −38% |
| Diversification ratio | ~2.4× |
| Lo 95% CI | excludes 0 ✅ |
| Block-bootstrap P5 | > 0 ✅ |
| Random-walk placebo | clears ✅ |
| CPCV / walk-forward | OOS positive, gap ~0.12–0.16 ✅ |
| Cost break-even | clears 2× the assumed cost ✅ |
| **Deflated Sharpe (honest n_trials)** | **❌ FAILS** — see below |

### ⚠️ The current honest verdict: Deflated Sharpe FAILS at the real trial count

This is the most important thing to know about this project's current state,
and it is exactly the kind of honesty the parent project lacked. On
**2026-07-15** `validation.honest_n_trials()` was corrected: the trial registry
that fed the Deflated-Sharpe calculation had only been counting **15** logged
trials, while the actual search log (`experiments.jsonl`) held **153** raw
config hashes. Unioning and de-duplicating both sources by effective config
gives the real count: **n=141 trials**.

| n_trials used | Deflated-Sharpe bar | Baseline net Sharpe (0.69) passes? |
|---:|---:|:---|
| 15 (stale registry) | ~0.53 | ✅ |
| 141 (honest, corrected) | **~0.72** | **❌ FAILS** |

Every other gate (noise floor, bootstrap, CPCV, walk-forward, cost-headroom)
passes — only the deflated-Sharpe hard gate fails, and it fails **because the
strategy space was searched more than the old bookkeeping admitted**, not
because a new bug was found. `docs/EDGE_GATE.md` / `docs/FINAL_RESEARCH_SUMMARY.md`
/ `todos.md` all carry this figure consistently.

**Consequences, currently in effect:**

- A **research moratorium** is in place: further backtest-only signal search is
  paused, because each additional evaluated config raises the deflated-Sharpe
  bar on the same edge (PBO across the 14 searched configs is **0.80**, well
  past the 0.5 overfit-warning threshold).
- The **long-history window** (1999–2026, ~27.5y) does clear H3 (deflated-max
  0.60 vs net 0.74 — more data lowers the bar) but is presented as reassuring
  context, not a reversal — the same config-search PBO caveat applies.
- The **forward (live/shadow) track is now the primary evidence channel** for
  anything new: see [Forward-testing & live paper trading](#forward-testing--live-paper-trading)
  below. Forward evidence pays no deflation tax.

**The ablation that set the defaults** (the discipline in action):

| Config | Net Sharpe | MaxDD | Calmar |
|---|---|---|---|
| baseline (equal weight, no governor) | 0.54 | −49% | 0.23 |
| + cluster weights | 0.49 ↓ | −50% | 0.19 ↓ |
| **+ governor (default)** | **0.65–0.69 ↑** | **~−36% ↑** | **0.34 ↑** |

Asset-class **cluster weighting was tested and DROPPED** (hurts Sharpe and
Calmar). The expanded 42-name universe and the combined `--ship-candidate`
preset were tested and **not promoted** — they win on a single 70/30 split but
lose on the honest walk-forward (see `docs/FINAL_RESEARCH_SUMMARY.md` for the
full decomposition). Financing (below) is the dominant real-world constraint on
any of these comparisons.

### H4 — does this beat doing nothing? (CAGR > MaxDD > Sharpe > Calmar)

Every gate above evaluates the edge in isolation. None of them ask the more
basic question: **is this worth doing at all, versus just buying SPY?** A
statistically real edge can still fail this — the headline Sharpe table above
is computed with **zero leverage cost**, and the whole diversification thesis
("IDM lets you lever a low-vol combined book to hit the vol target") only
turns a Sharpe advantage into a CAGR advantage if that leverage is actually
free, which it never is.

`edge_gate.py`'s **H4 `beats_buy_hold`** gate checks this directly, and — this
is the important part — on the priority order **CAGR > MaxDD > Sharpe >
Calmar**, not Sharpe-first: it passes if CAGR beats SPY outright, *or* CAGR is
within 2 points of SPY's *and* max drawdown is at least 10% smaller in
magnitude. A real MaxDD win doesn't rescue a big CAGR shortfall (a hard 1x or
3x leverage cap fails this even with much better drawdowns — see below); a
close-enough CAGR with a meaningfully safer drawdown does pass, even if Sharpe
trails.

Swept across leverage levels, all at the 1% financing rate this project's own
docs already call realistic:

| Gross cap | CAGR | MaxDD | Sharpe | vs. SPY (11.11% CAGR, −55.2% MaxDD, 0.63 Sharpe) | H4 |
|---|---|---|---|---|---|
| 1.0x | 3.08% | −15.1% | 0.49 | far short on CAGR | ❌ FAIL |
| 3.0x | 7.57% | −31.6% | 0.56 | still short on CAGR, even with much better MaxDD | ❌ FAIL |
| **Uncapped (~4x, the validated default)** | **10.26%** | **−41.9%** | **0.57** | **CAGR close, MaxDD decisively better** | **✅ PASS** |

The counter-intuitive result: capping leverage to control risk (1x, 3x) makes
this **fail** H4, because it gives up too much CAGR for a drawdown improvement
that isn't worth as much on this priority order. The uncapped configuration —
this book's natural gross exposure, honestly financed — is the one that
clears it, by nearly matching the index's return while drawing down ~13
points less.

This gate is why `validated_config()` (below) charges financing but does
**not** cap gross notional, and why the real paper account's leverage was
raised to match.

---

## §carry — the data-honesty note (the §84 lesson, applied from day one)

Real carry needs the **futures term structure** (front vs deferred contract),
and proper continuous back-adjusted futures need a **paid feed** (CSI /
Norgate / a broker). The free ETF-proxy path cannot fully express true carry:

- `rules.carry_forecast()` is fully implemented and unit-tested. `--carry`
  wires a synthetic series for demonstration; `--carry-proxies` wires a free
  FRED/dividend-yield proxy; `--real-bond-carry` wires tenor-specific
  Treasury-curve roll-down carry (still a proxy, not a real futures curve).
- The default trend(+COT)-only book runs clean on free data and alone
  demonstrates the diversification edge.
- Populating carry with real term-structure data remains the first paid-data
  upgrade (the `Instrument.kind="future"` / `carry_kind` fields already
  anticipate it).

This is the same discipline that was missing before: name the data limit up
front, don't fake the result.

---

## Forward-testing & live paper trading

This is the part the parent project never had: an automated nightly loop that
runs the validated config against real closes, tracks a no-broker shadow book,
reconciles live-vs-backtest, replay-checks every stored decision for code-level
drift, and (once shadow tracking was confirmed) submits real paper orders to a
dedicated Alpaca account.

### The nightly loop (`scripts/forward_loop.sh`, launchd-scheduled)

Runs daily (currently 3:00 PM PT / 6:00 PM ET — ~2h after the US close, giving
price data time to settle without an excessive delay):

1. **`warm_cache.py`** — refreshes the point-in-time price cache. yfinance
   `auto_adjust=True` re-adjusts the *entire* history at every ex-dividend, so a
   naive refresh would silently rewrite the past the engine already traded on.
   The cache instead keeps every already-cached date **verbatim** and
   ratio-stitches new rows onto that basis (`data.py`); rejected upstream
   revisions are logged, not applied. The COT cache uses the same shared
   point-in-time merge primitive (`_pit_merge`, `data.py`/`cot_data.py`).
2. **`generate_targets.py`** — writes today's target position for three
   parallel books, all from the shared config-generation path:
   - **champion** — the validated default (core 19 + governor + 30% buffer + COT).
   - **challenger** — champion with the COT lever flipped, to forward-test it
     against champion head-to-head.
   - **challenger_semis** — champion + the SMH/SOXX/XSD semis pack, forward-testing
     a lever that ranked well on backtest but under a PBO=0.80 caveat.
3. **`shadow_book.py`** — marks each book's next-day return with **no broker
   costs** (a pure "did the model's math hold" check), independent of whether
   any real order was ever submitted.
4. **`reconcile.py`** — compares live vs modeled returns (gross for the shadow
   book, net for the real broker path), reports correlation/tracking-error/drift,
   a Perold-style drift decomposition (α / β-gap / residual), a worst-quartile
   edge-decay flag, and an **input-revision check** — recomputes recent targets'
   forecasts from today's data and flags any symbol whose stored value no
   longer matches (catches *data* changing under a stored decision).
5. **`detect_drift.py --enforce`** — Phase-3 **replay-based decision drift
   detection**: re-derives every stored snapshot's decision from its
   point-in-time feature snapshot and classifies any mismatch as `matched`
   (healthy), `data` (the price/COT history was revised upstream — benign),
   `lineage` (the data-handling regime changed), or `logic` (same inputs,
   different output ⇒ the code changed — this is the alarming case). Only
   `logic` drift engages the **kill switch** (`data/kill_switch.json`); it does
   not auto-clear on a later clean run — clearing it is always a human,
   documented decision.
6. **`execute_alpaca.py`** — submits delta-notional paper orders (a *dedicated*
   Alpaca paper account, isolated from any other trading system so positions on
   overlapping tickers never net together) to match the **champion** book only —
   the broker never trades a challenger. Respects the kill switch. Runs an
   optional AI pre-trade evaluation (below) before sizing. `--max-gross-mult`
   caps submitted gross notional at that multiple of account equity/cash
   (`--use-cash-balance` sizes against cash so the long leg is fully cash-paid;
   short legs still draw Reg-T margin regardless). Raised from **1.0x to 4.0x**
   on 2026-07-22 to match both `validated_config()`'s natural gross exposure
   (median ~3.9x) and this account's real Reg-T buying power (~4x equity) — a
   1.0x cap had the real account running at roughly a quarter of the modeled
   book's risk, which is why the daily "modeled Sharpe" never matched what was
   actually held (see the H4 section above for why ~4x, not 1x or 3x, is the
   leverage level that actually clears the buy-and-hold check).

### AI pre-trade evaluator (`evaluator.py`, on by default in `execute_alpaca.py`)

Before submitting orders, an LLM call (default: **Kimi / Moonshot AI**, via its
OpenAI-compatible endpoint; a generic OpenAI-compatible provider is also
supported) reviews the proposed book and returns a confidence/reasoning/scale
verdict. It's an **advisory execution overlay**, not a replacement for the
validated engine:

- `--ai-mode advisory` — logs reasoning only, no effect on sizing.
- `--ai-mode scale` (default) — multiplies the whole book by the returned scale
  factor (0–1).
- `--ai-mode block` — skips the rebalance entirely on a `reject` verdict (falls
  back to no-op unless `--ai-required`).

Falls back to a no-op evaluator (approve, scale=1.0) if no API key is
configured. Every call is logged to `data/ai_evaluations.jsonl`.

### Promotion framework (`edge_gate.py`)

A candidate config (or a forward challenger book) only gets promoted to the
default if it clears a structured gate, not a vibe:

- **Hard gates** (any failure ⇒ overall FAIL): clears the random-walk noise
  floor, block-bootstrap 5th-percentile Sharpe > 0, the **honest** Deflated
  Sharpe (see above — this is the one currently failing for the base edge), and
  **H4 beats_buy_hold** — CAGR/MaxDD/Sharpe vs. a trivial SPY buy-and-hold, on a
  CAGR > MaxDD > Sharpe > Calmar priority rather than Sharpe-first (see the H4
  section above).
- **Robustness gates** (pass required for PASS, not just CONDITIONAL): CPCV
  robustness, purged walk-forward OOS > 0 with a bounded IS/OOS gap, and cost
  headroom ≥ 2× the assumed cost.
- **Forward-evidence gate**: a challenger book is only promoted if it has
  **won** — not merely survived — `champion_challenger_report` (≥60 forward
  days *and* beating champion's Sharpe by a minimum margin), and the comparison
  set's PBO must be ≤0.5. "Days accrued alone are not evidence... a challenger
  that merely survives 60 days may have lost throughout" (`edge_gate.py`
  docstring).

### Artifacts

| File | Purpose |
|------|---------|
| `data/live_targets.jsonl` | Date-stamped target units/notional/forecast/config per book. |
| `data/live_returns.csv` | Realised daily shadow/live returns. |
| `data/reconciliation/YYYY-MM-DD.json` | Daily correlation, tracking error, drift, edge-decay, input-revision report. |
| `data/feature_snapshots/` | Point-in-time feature snapshots Phase-3 replay re-derives decisions from. |
| `data/kill_switch.json` | `{"paused": true, ...}` with a documented root cause when a guardrail fires. |
| `data/broker_orders.jsonl` | Submitted Alpaca paper orders. |
| `data/ai_evaluations.jsonl` | AI pre-trade evaluation records. |
| `data/price_revisions.jsonl`, `data/cot_revisions.jsonl` | Rejected upstream data revisions (PIT cache defence). |

---

## Monitoring & flag taxonomy

`--help` groups ~70 flags into:

- **CORE** (shape the validated default): `--vol-target --buffer --cost-scheme
  --weight-scheme --no-governor --governor-smooth`
- **VALIDATED-POSITIVE** (opt-in, clears the walk-forward):
  - `--cot` — free CFTC Commitments-of-Traders positioning, contrarian sign.
    Full-history 0.69→0.72, walk-forward OOS 0.61→0.63. Forward-testing via the
    `challenger` book (A/B against champion) before promotion.
  - `--network-momentum` — price-only lead-lag graph momentum, a fourth
    orthogonal rule alongside EWMAC/breakout. Full 0.69→0.73, WF OOS 0.63→0.67.
    **Shipped.**
  - `--semis` — adds SMH/SOXX/XSD. WF OOS 0.63→0.69. Forward-testing via the
    `challenger_semis` book (PBO=0.80 caveat on the backtest ranking).
  - `--qqq` — adds QQQ. WF OOS 0.63→0.68.
- **RESEARCH** (tested, none beat the walk-forward, opt-in only): `--cluster-weights
  --expanded-universe --empirical-scalars --regime-overlay --vix-term-overlay
  --credit-overlay --hmm-regime-overlay --equity-momentum-sleeve
  --curve-steepener --real-bond-carry --garch-vol --accel --xsmom --corr-spike
  --carry-proxies --core-commodities --cot-momentum --ship-candidate
  --drawdown-control --trend-strength-filter --calibration-smooth`
- **INSTRUMENT PACKS** (leverage-dependent — compare only with `--financing-rate`
  set): `--crypto` (BTC-USD/ETH-USD, risk-weight capped), `--curated-breadth`
  (10 correlation-selected names), `--diversifier-pack --rate-pack` (bond/credit
  ETFs — win on paper, shrink hard under realistic financing).
- **VALIDATION / DIAGNOSTICS**: `--validate --oos --walk-forward --diagnostics
  --monitor --n-trials --placebo --alarm-on-worst-quartile`

Notes on the leverage-dependent packs: **`--financing-rate`** charges an annual
spread on gross notional above `--financing-threshold` (default 1.0×). Without
it, levered low-vol bond packs look like a free Sharpe improvement — 1%
financing cuts baseline Net Sharpe roughly 0.65→0.54 and reshuffles the
best-looking variants toward equity packs (`docs/FINANCING_AND_LEVERAGE.md`).
**`--max-gross N`** applies a hard gross-notional cap for the same
reality-check. A parked **VRP** (variance-risk-premium) data layer
(`vrp_data.py`) exists but is not wired to the CLI — injecting a fat-tailed
short-vol stream as a tradable instrument breaks the engine's vol-targeting;
harvesting it needs a dedicated position-capped sizing path, not instrument
injection.

---

## Roadmap

1. Real futures term-structure feed → genuine carry across all asset classes
   (the paid-data upgrade; the known-honest path to a materially bigger edge
   now that the free-lever search is paused).
2. Resolve the forward challengers: COT (`challenger`) and the semis pack
   (`challenger_semis`) both need ≥60 forward days and to **win**, not just
   survive, `champion_challenger_report` before promotion.
3. A correlation/Sharpe-aware instrument weighting scheme (cluster weighting
   was tried, hurt, and was dropped — this is the follow-up, still research).
4. Multiple-contract forecast mapping & roll handling, once a real futures feed
   exists.
5. ~~Live execution layer~~ — **shipped**: dedicated Alpaca paper account, kill
   switch, AI pre-trade evaluator, replay-based drift detection. Live (real
   capital) trading remains gated on an extended, clean paper-trading record.
6. ~~Validate against a trivial buy-and-hold benchmark~~ — **shipped**
   (2026-07-22): the H4 gate above, and `validated_config()`/the real paper
   account's leverage corrected to the configuration that actually clears it.
   Worth re-checking periodically — H4 currently passes on a close-CAGR/
   better-MaxDD basis, not by a wide margin, so it's not a permanently settled
   question.
