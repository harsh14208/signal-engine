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
Mean pairwise correlation          : ~0.00–0.02  (low; idiosyncratic noise dominates)
Portfolio Sharpe                   : 0.57
Diversification ratio              : 4.7×   ← the whole thesis
Deflated Sharpe (100 trials)       : 0.57 < 0.77  ⚠ FAILS  (the tooling has teeth)
Random-walk placebo 95th pct       : 0.39  → real 0.57 ✅ clears the floor
```

It is **not** a claim about live performance — that comes from `--source yfinance`
on real history (below). The honesty tooling is the product; the synthetic number
is a fixture.

### Real-data results (`--source cache`, 2007–2026, 19 ETF proxies)

Default config (trend-only, equal-weight, realised-vol governor ON, 30% buffer) on
~4,900 days of actual prices:

| Metric | Value |
|---|---|
| **Net Sharpe** | **0.68** (gross 0.72) |
| Realised vol | 21.1% (vs 20% target) |
| Max drawdown | −36.3% |
| Calmar | 0.35 |
| Diversification ratio | 2.4× (mean standalone → portfolio 0.68) |
| Lo 95% CI | [0.23, 1.13] — SR=0 outside ✅ |
| Block-bootstrap P5 | 0.34 > 0 ✅ |
| Random-walk placebo | clears (0.68 vs 0.37 floor) ✅ |
| **Deflated Sharpe (16 _real_ trials)** | **0.68 > 0.54 ✅ clears at the honest trial count** |
| Single 70/30 split IS/OOS | 0.75 / 0.51 — gap +0.24 ⚠ |
| **Walk-forward (4-fold) mean OOS** | **0.59 — gap +0.14** ← the honest test |

**Read the walk-forward, not the single split.** A single 70/30 cut puts the entire
hold-out in the choppy 2020–2026 window (OOS 0.51, gap +0.24 — pessimistic). The
4-fold purged walk-forward — the honest multi-period test — gives **mean OOS 0.59,
gap +0.14**. The Deflated Sharpe now uses the *actual logged* trial count (16, via
`experiments.py`), not a placeholder, and the edge **clears it**; SR=0 is outside the
Lo CI; it clears the random-walk placebo. This is a real ~0.6 out-of-sample edge.

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

### `--ship-candidate` and overlays — a single-split false dawn (kept honest)

The `--ship-candidate` preset (expanded 42-ETF universe + regime overlay + 30%
buffer) looks like a big win **on a single 70/30 split**: net SR 0.74, OOS **0.72**,
gap +0.03. It is **not** a validated improvement — the walk-forward refutes it:

| Config | Full Sh | Calmar | Divers | single-split OOS | **walk-fwd mean OOS** | WF gap |
|---|---|---|---|---|---|---|
| **default** (core 19) | 0.68 | 0.35 | 2.4× | 0.51 | **0.59** | **+0.14** |
| + expanded universe (42) | 0.69 | 0.35 | 2.9× | 0.52 | 0.53 ↓ | +0.29 |
| + regime overlay only | 0.69 | 0.36 | 2.4× | 0.54 | 0.59 | +0.14 |
| ship-candidate (both) | 0.74 | 0.40 | 2.9× | **0.72** | **0.54 ↓** | +0.28 |

On the honest multi-fold test the ship-candidate's OOS (0.54) is **worse** than the
plain default (0.59) with double the gap — its 0.72 single-split OOS was just the
2020–2026 window flattering the expanded+regime combo. Decomposition: the **expanded
universe** lifts the diversification *ratio* and IS Sharpe, but the younger/thinner
ETFs over-fit IS (walk-forward OOS drops) **and** raise the placebo floor 0.37→0.56,
so signal-to-noise actually *worsens* (0.74/0.56 = 1.3× vs the default's 0.68/0.37 =
1.8×). The **regime overlay** is ~inert on the walk-forward.

So `--ship-candidate` is a **research flag, not promoted** — the validated default
(core 19, governor) wins on the honest test. The `--vix-term-overlay`
(`^VIX9D/^VIX`, `^VIX3M/^VIX` term structure) and `--credit-overlay` (FRED `BAA10Y`)
were likewise tested and **left opt-in** (no walk-forward improvement). The whole
project's discipline in one line: **promote on the walk-forward, never the single
split.**

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

## Monitoring & flag taxonomy

- **`--cot` (VALIDATED-POSITIVE):** free CFTC Commitments-of-Traders positioning is
  the **first free signal to clear the walk-forward** (full 0.69→0.72, walk-forward
  OOS 0.61→0.63, pre-specified contrarian sign). It's a per-instrument *rule* on ~10
  macro-core ETFs combined via FDM — orthogonal to price (it's *positioning*), unlike
  every overlay that came before it. Kept opt-in (needs a network fetch + the margin is
  modest and fold-concentrated); recommended for promotion pending wider coverage.

- **`--network-momentum` (VALIDATED-POSITIVE):** a price-only lead-lag graph signal
  (follow the leader cross-sectional momentum). It adds a fourth orthogonal rule to
  EWMAC/breakout, clears the walk-forward with no leverage increase, and needs no new
  instrument data. Full sample 0.69→0.73, walk-forward OOS 0.63→0.67. **Shipped** as a
  validated rule toggle.


- **`--monitor`** prints the strategy's rolling 1-year Sharpe with an edge-decay
  alarm; `monitor.reconcile(live, backtest)` scores live-vs-backtest agreement
  (correlation / tracking error / drift) for when live returns exist — the
  reconciliation harness the parent project never had. Recent additions: a Perold-
  style drift decomposition (α / β-gap / residual) and a worst-quartile edge-decay
  flag (`--alarm-on-worst-quartile`).
- **`--semis` (VALIDATED-POSITIVE):** adds SMH/SOXX/XSD to the core universe.
  Walk-forward OOS improves from 0.63 to 0.69, traded days increase, and gross
  exposure stays in line with the baseline. **Shipped** as a validated instrument-pack
  toggle.

- **`--qqq` (VALIDATED-POSITIVE):** adds QQQ (Nasdaq-100) to the core universe.
  Walk-forward OOS improves from 0.63 to 0.68 with no leverage blow-up. **Shipped** as
  a validated instrument-pack toggle.

- **`--diversifier-pack`, `--rate-pack`** add cross-asset bond/credit/commodity
  ETFs (BNDX/PFF/AMLP/MUB/EMLC and BNDX/MUB). They produce the highest uncapped
  Sharpe but depend on levering low-vol bonds. Left as opt-in instrument packs; pair
  with `--max-gross` to reality-check implementation feasibility.
- **`--max-gross N`** applies a gross-notional exposure cap (e.g. `3.0` for 3×
  capital) post-governor. Useful for reality-checking how much an apparent edge
  depends on levering low-vol instruments; note that a tight cap lowers realized
  vol and can push Sharpe below the Deflated-Sharpe bar.
- **`--financing-rate R`** charges an annual spread on gross notional above
  `--financing-threshold` (default 1.0). Without this, levered bond packs look like
  a free Sharpe improvement. See `docs/FINANCING_AND_LEVERAGE.md` — the 1% case
  reduces Sharpe by ~0.10–0.13 and reshuffles the best variants toward equity packs.
- **Research-only overlays (diagnostic / opt-in):** drawdown-state control
  (`--drawdown-control`), trend-strength filter (`--trend-strength-filter`), and
  calibration smoothing (`--calibration-smooth`) are implemented but did **not**
  improve walk-forward OOS Sharpe versus the financed baseline. They remain available
  for experimentation. See `docs/OPTIMIZATIONS.md`.
- **`--help` ends with a flag taxonomy**: CORE (validated) / VALIDATED-POSITIVE
  (clears walk-forward, safe to ship) / RESEARCH (tested, none beat default) /
  INSTRUMENT PACKS (leverage-dependent) / VALIDATION-DIAGNOSTICS. ~70 flags exist
  but only a handful shape the validated default.
- **VRP note:** a free VRP data layer exists (`vrp_data.py`, CBOE vol indices —
  no options panel needed) but is **parked**: injecting a fat-tailed short-vol
  stream as a tradable instrument detonates the engine's vol-targeting. Harvesting
  VRP needs a dedicated position-capped sizing path, not instrument injection.

## Roadmap (only after real-data OOS + placebo pass)

1. Real futures term-structure feed → genuine carry across all asset classes.
2. ~~Asset-class cluster weights~~ — *tried, hurt, dropped* (see ablation). Next:
   a **correlation/Sharpe-aware** weighting (not asset-class), plus **governor
   smoothing** to cut its added turnover (47x → 61x).
3. Multiple-contract forecast mapping & roll handling.
4. Live execution layer (broker, position reconciliation) — **last**, not first.
