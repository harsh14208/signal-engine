# Long-history backtest — deployed config, 1999–2026

Reproduce:
```bash
# one-time: fetch the max-history panel (writes data/prices_long.parquet, gitignored)
python -c "from signal_engine.data import load_prices; from signal_engine.markets import symbols; \
          load_prices(symbols(), start='1999-01-01', source='yfinance', cache_tag='long')"
# then run the full honesty battery over max history:
python scripts/validate_edge.py --long --source cache
```

## Data honesty (read first)
The panel spans **1999–2026 (27.5y)** but the universe is ETF proxies with ragged
inception, so the *diversified* thesis only fully exists from 2007:

| Year | Instruments live |
|---|---|
| 2000 | 3 / 19 (SPY, EWJ, IWM) |
| 2004 | 11 / 19 |
| **2007+** | **19 / 19 (full basket)** |

Treat pre-2007 as indicative, not a fair test. And it's still ETF proxies, not
real futures.

## Deployed config (COT + governor + 30% buffer) — by era

| Era | Sharpe | Vol | Return | MaxDD |
|---|---:|---:|---:|---:|
| **Full 1999–2026** | **0.74** | 20.6% | 15.3% | −38% |
| Full basket 2007–26 | 0.74 | 21.5% | 15.8% | −38% |
| GFC 2007–09 | 0.88 | 21.3% | 18.7% | −16.8% |
| 2010s | 0.68 | 21.3% | 14.4% | −26.4% |
| COVID 2020 | 1.93 | 22.4% | 43.1% | −14.9% |
| 2021–22 (inflation/bear) | 0.86 | 23.1% | 20.0% | −18.2% |
| **2023–26 (recent)** | **0.34** | 21.0% | 7.1% | −30.6% |

~35× cumulative equity. Classic trend crisis alpha (GFC/COVID/2022 strong); recent
years weak, consistent with the industry-wide 2023–25 trend drawdown.

## Honesty battery over the full 27.5y — ✅ PASS (comfortable)
```
H1 clears_noise:    net 0.74 vs noise-floor 0.40   ✅
H2 edge_real:       bootstrap P5 0.46 > 0          ✅
H3 passes_deflated: net 0.74 vs deflated-max 0.58  ✅  (clears by 0.16, not 0.02)
R1 cpcv_robust:     OOS P5 0.39, 0% paths<0        ✅
R2 walk_forward_ok: OOS 0.72, gap -0.00            ✅  (~zero IS/OOS degradation)
R3 cost_headroom:   net positive past 25bps        ✅
effective bets 13.9 / 19
```
The longer window lowers the deflated bar (more statistical power: N=6921 vs 4901),
so H3 clears far more comfortably than on the 2007–26 window (0.71 vs 0.69).

## The honest synthesis
1. **Robust in-sample:** 0.74 over 27 years, near-zero walk-forward gap, genuine
   crisis alpha. Far more defensible than the vanilla baseline (which *fails*
   deflation at n=100 — see `experiment_results.md`).
2. **But selected on this data:** the deployed config (which levers, buffer=0.30)
   was chosen because it looked best here. PBO=0.80 across the lever search warns
   that this selection is itself overfit. Walk-forward mitigates but can't rule out
   fitting the whole config family to this specific 27-year path.
3. **Unavoidable caveats:** ETF proxies (not real futures); 2023–26 weak (possible
   edge decay).

PBO is a *multi-config* metric (config-search overfitting), so it doesn't apply to
the single deployed config — CPCV (OOS P5 0.39, robust) is the right per-config
test, and it passes.

**Verdict:** good enough to keep the deployed config as the live candidate; not
good enough to call proven. The one test that can't be overfit is still ahead:
forward (live) data on real futures. The near-zero walk-forward gap is the best
in-sample evidence available — everything beyond it needs out-of-sample time.
