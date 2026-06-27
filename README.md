# signal-engine

A diversified systematic **futures trend + carry** engine (Carver / AHL style).

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
   a tiny, honest data layer with a deterministic synthetic generator so the
   whole pipeline is verifiable offline, and an explicit data-honesty note for
   carry (see §carry).
5. **No crisp edge thesis (kitchen sink).** → *Here:* one sentence, at the top of
   this file, that every line of code serves.

**Build discipline:** research-harness *first*. There is intentionally **no live
broker / execution layer yet** — that gets written only after the edge survives
real-data OOS + placebo. That sequencing is the whole point.

---

## How it works

```
prices ─▶ returns ─▶ blended vol ─▶ rule forecasts ─▶ combine (FDM)
       ─▶ vol-target sizing (IDM) ─▶ no-trade buffer ─▶ shift(1) ─▶ P&L − costs
```

| Concept | Module | What it does |
|---|---|---|
| Volatility | `volatility.py` | Carver blended EW vol (70% recent / 30% long-run) — the denominator that lets gold and bonds speak one language |
| Trend rules | `rules.py` | EWMAC crossover at 3 speeds + breakout, vol-normalised, scaled to mean \|f\|≈10, capped ±20 |
| Carry rule | `rules.py` | Risk-adjusted annualised carry (term-structure driven — see §carry) |
| Combine | `forecast.py` | Weighted sum × **FDM** (Forecast Diversification Multiplier) |
| Sizing | `portfolio.py` | Volatility targeting + **IDM** (Instrument Diversification Multiplier) — *this is where diversification becomes return* |
| Turnover | `portfolio.py` | No-trade buffer band (expanding, no lookahead) |
| Engine | `backtest.py` | No-lookahead P&L (positions decided at close *t-1*) net of costs |
| Metrics | `metrics.py` | Sharpe, MaxDD, CAGR, Calmar, Sortino, skew, turnover |
| **Rigor** | `validation.py` | Lo (2002) CI · **Deflated Sharpe** · block-bootstrap MC · **random-walk placebo** — ported from the parent project |
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

# Chronological in-sample / out-of-sample split
python -m signal_engine --oos 0.7 --validate

# Real ETF-proxy data (needs network + `pip install -e .[data]`)
python -m signal_engine --source yfinance --validate
```

Tests:

```bash
pip install -e .[dev]
pytest            # 40 tests
ruff check signal_engine tests
```

### What the synthetic demo shows (and does NOT claim)

The synthetic generator (`data.synthetic_prices`) is a **labelled test/demo DGP**
with persistent trends and low correlation — it exists to prove the pipeline and
the diversification *math* are correct and verifiable offline. A representative
run:

```
Mean standalone instrument Sharpe : 0.12   (no single bet is impressive)
Mean pairwise correlation          : ~0.00
Portfolio Sharpe                   : 0.57
Diversification ratio              : 4.7×   ← the whole thesis
Deflated Sharpe (100 trials)       : 0.57 < 0.77  ⚠ FAILS  (the tooling has teeth)
Random-walk placebo 95th pct       : 0.39  → real 0.57 ✅ clears the floor
```

It is **not** a claim about live performance — that comes from `--source yfinance`
on real history (below). The honesty tooling is the product; the synthetic number
is a fixture.

### Real-data results (`--source cache`, 2007–2026, 19 ETF proxies)

Default config (trend-only, equal-weight, realised-vol governor ON) on ~4,900 days
of actual prices:

| Metric | Value |
|---|---|
| **Net Sharpe** | **0.65** (gross 0.69) |
| Realised vol | 21.2% (vs 20% target) |
| Max drawdown | −36.2% |
| Calmar / Sortino | 0.34 / 0.88 |
| Diversification ratio | 2.2× (mean standalone 0.25 → portfolio 0.65) |
| Lo 95% CI | [0.20, 1.10] — SR=0 outside ✅ |
| Block-bootstrap P5 | 0.31 > 0 ✅ |
| Random-walk placebo | clears (0.65 vs 0.42 floor) ✅ |
| IS / OOS (70/30) | 0.71 / 0.51 — gap +0.20 ⚠ |
| Deflated Sharpe (100 trials) | 0.65 < 0.69 ⚠ (clears at this project's real handful of trials) |

**Honest caveats:** the IS/OOS gap is wide — the governor's vol-targeting paid off
more in the cleaner-trending 2007–2019 era than the choppy 2020–2026 hold-out
(OOS 0.51 is still positive and within bootstrap noise of the ungoverned 0.57). It
has **no fitted parameters**, so this is regime, not overfitting. The 100-trial
Deflated bar is a conservative placeholder; at this project's actual handful of
configs the bar (~0.49) is cleared.

**The ablation that set the defaults** (the discipline in action):

| Config | Net Sharpe | MaxDD | Calmar | OOS |
|---|---|---|---|---|
| baseline (equal, no governor) | 0.54 | −49% | 0.23 | 0.57 |
| + cluster weights | 0.49 ↓ | −50% | 0.19 ↓ | 0.42 ↓ |
| **+ governor (default)** | **0.65 ↑** | **−36% ↑** | **0.34 ↑** | 0.51 |

Asset-class **cluster weighting was tested and DROPPED**: equal-per-cluster
overweights singleton/small clusters holding weak names (the lone −0.11-Sharpe
REIT, the 2-name credit sleeve). Shipping a plausible-but-harmful knob is the exact
mistake the parent project made 96 times — here it's a one-line research flag
(`--cluster-weights`), off by default. The **governor** is the validated win.

---

## §carry — the data-honesty note (the §84 lesson, applied from day one)

Real carry needs the **futures term structure** (front vs deferred contract), and
proper continuous back-adjusted futures need a **paid feed** (CSI / Norgate / a
broker). The free ETF-proxy path cannot express true carry, so:

- `rules.carry_forecast()` is fully implemented and unit-tested against a carry
  series, and `--carry` wires a *synthetic* series for demonstration only.
- The default trend-only book runs clean on free data and alone demonstrates the
  diversification edge.
- Populating carry with real term-structure data is the first paid-data upgrade
  (the `Instrument.kind="future"` / `carry_kind` fields already anticipate it).

This is the same discipline that was missing before: name the data limit up
front, don't fake the result.

---

## Roadmap (only after real-data OOS + placebo pass)

1. Real futures term-structure feed → genuine carry across all asset classes.
2. ~~Asset-class cluster weights~~ — *tried, hurt, dropped* (see ablation). Next:
   a **correlation/Sharpe-aware** weighting (not asset-class), plus **governor
   smoothing** to cut its added turnover (47x → 61x).
3. Multiple-contract forecast mapping & roll handling.
4. Live execution layer (broker, position reconciliation) — **last**, not first.
