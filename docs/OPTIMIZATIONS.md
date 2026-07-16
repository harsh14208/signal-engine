# Additional optimizations implemented

This document tracks the six optimizations identified from `RESEARCH_LEVERS.md`, `PATENT_PRIOR_ART.md`, and the long-history weakness in `LONG_HISTORY.md`.

## 1. Implementation-shortfall decomposition of drift

**Source:** `PATENT_PRIOR_ART.md` (Perold TCA).

The reconciliation report now exposes the live-vs-model drift decomposition that was already computed by `monitor.decompose_drift`:

- `alpha` — annualised unexplained drift (real costs, data lag).
- `beta_gap` — drift from running a different exposure than modeled (`β ≠ 1`).
- `residual` — annualised tracking noise.
- `total_drift` — sum of alpha + beta_gap, reconstructing the headline drift.

`scripts/reconcile.py` prints these under "Drift decomposition". A Perold-style helper `monitor.implementation_shortfall_components` is available for when arrival prices are supplied.

**Status:** shipped. Pure diagnostic, no overfit risk.

## 2. Warm-up / stateful-restart parity

**Source:** `PATENT_PRIOR_ART.md` (QuantConnect restart divergence).

Added `Config.min_history_required()` and a guard in `live.generate_target()` that raises a clear error if the available price history is shorter than the longest lookback required by the enabled rules/overlays. This prevents a cold-started loop from silently producing different indicator state than a continuously-running loop.

**Status:** shipped. Safety guard.

## 3. Quartile-based edge-decay flag

**Source:** `RESEARCH_LEVERS.md` / `PATENT_PRIOR_ART.md` (arXiv 2604.18821).

`monitor.edge_decay_report` already reported `worst_quartile`; it can now drive the kill switch via `alarm_on_worst_quartile=True`. `scripts/reconcile.py` accepts `--alarm-on-worst-quartile`.

- Absolute floor (`alarm_floor`) catches catastrophic decay.
- Worst-quartile flag catches earlier, relative decay without hardcoding a Sharpe level.

**Status:** shipped. Opt-in via CLI.

## 4. Weight persistence across recalibration

**Source:** `RESEARCH_LEVERS.md`.

Added `Config.calibration_smooth`. When set (e.g., `20`), the expanding-window transitions for instrument weights, IDM, and FDM are linearly blended over N days instead of jumping instantly at each rebal date. This cuts estimation-driven turnover.

**CLI:** `--calibration-smooth N`

**Status:** shipped. Test confirms reduced turnover with similar realised vol.

## 5. Drawdown-state control overlay

**Source:** `RESEARCH_LEVERS.md`.

Added an opt-in overlay that scales positions down when the strategy hits a realised drawdown threshold and scales back up on recovery.

**Config:**
- `use_drawdown_control`
- `drawdown_threshold` (default 10%)
- `drawdown_scale` (default 50%)
- `drawdown_recovery` (default 5%)

**CLI:** `--drawdown-control`, `--drawdown-threshold`, `--drawdown-scale`, `--drawdown-recovery`

**Status:** shipped. Causal; applied post-governor. Keep opt-in because thresholds are easy to overfit.

## 6. Trend-strength filter (addressing 2023–26 weakness)

**Source:** `LONG_HISTORY.md` (Sharpe dropped to 0.34 in 2023–26 vs 0.74 full sample).

Added an opt-in filter that de-gears the book when the average absolute combined forecast falls into the bottom percentile of its recent history. This targets low-trend regimes.

**Config:**
- `use_trend_strength_filter`
- `trend_strength_window` (default 63)
- `trend_strength_threshold` (default 0.25 = bottom quartile)
- `trend_strength_scale` (default 70%)

**CLI:** `--trend-strength-filter`, `--trend-strength-window`, `--trend-strength-threshold`, `--trend-strength-scale`

**Status:** shipped. Must be validated OOS; default parameters are a starting point, not a tuned solution. The 2023–26 weakness may also be genuine industry-wide trend decay that no in-sample filter can fix.

## Files changed

- `signal_engine/config.py` — new fields.
- `signal_engine/backtest.py` — wiring for trend-strength, drawdown, calibration smoothing.
- `signal_engine/portfolio.py` — `drawdown_overlay`, `trend_strength_overlay`.
- `signal_engine/monitor.py` — enhanced drift decomposition.
- `signal_engine/live.py` — history guard, `alarm_on_worst_quartile` passthrough.
- `signal_engine/cli.py` — new flags + restored missing `--network-momentum` flag.
- `scripts/reconcile.py` — prints drift decomposition, `--alarm-on-worst-quartile`.
- `tests/test_backtest.py`, `tests/test_monitor.py`, `tests/test_live_pipeline.py`, `tests/test_cli.py` — coverage.
