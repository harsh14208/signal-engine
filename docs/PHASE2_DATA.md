# Phase 2 — data robustness (calendar, quality, feature store, lineage)

Adapts TRS's DB-backed data infrastructure to signal-engine's file-based, offline,
deterministic idiom. Four modules, no new hard dependency.

## NYSE market calendar — `signal_engine/market_calendar.py`
Expected trading sessions so a **data gap** is distinguishable from a holiday.
Built on pandas' holiday primitives + an `AD_HOC_CLOSURES` list (9/11, Hurricane
Sandy, national days of mourning) that a rules calendar can't derive.

```python
mc.trading_days(start, end)        # expected NYSE sessions
mc.missing_sessions(price_index)   # sessions expected but absent = candidate gaps
mc.is_trading_day / next_trading_day / previous_trading_day
mc.sessions_until_next_holiday(date)   # for a pre-holiday liquidity haircut
```
Caveat: `nearest_workday` observance is the standard approximation (~±1 session/yr);
good for gap detection, not settlement-grade.

## Data-quality audit — `signal_engine/data_quality.py`
The file-based analogue of TRS's provider-reliability scorecard: it scores the
**panel** (the real risk for a backtest), not a live API.

```bash
python scripts/audit_data.py --source cache
```
Per symbol: internal gaps (vs calendar), staleness, extreme jumps (unadjusted
splits / bad prints), flatline runs (dead feed) → a 0–100 health score. Panel report
flags symbols below threshold and lists missing sessions. Exit 1 if unhealthy.
`record_fetch` / `health_scorecard` give a live-path fetch-telemetry log.

**On your real cache:** mean health 95/100, no symbols flagged; the only "missing"
session is today's not-yet-fetched bar. (Encoding the ad-hoc closures removed the
Sandy/mourning-day false positives.)

## Point-in-time feature store — `signal_engine/feature_store.py`
Immutable, content-hashed snapshots of the exact decision inputs, stamped with the
data lineage. Config snapshots rebuild the *model*; feature snapshots replay the
*decision* — the substrate Phase 3's replay-drift detection needs.

```python
save_snapshot(as_of, features, book="champion")   # immutable: conflicting rewrite raises
load_snapshot(as_of, book); list_snapshots()
```
`generate_target(..., snapshot=True)` now writes one per target automatically.

## Lineage — `signal_engine/lineage.py`
Versioned provenance (vendor, adjustment mode, calendar, universe hash, schema
versions) → `lineage_hash()`, stamped into every target record and snapshot. When
live diverges, "did the data lineage change?" becomes answerable.

## Integration
- Target records carry `lineage_hash`; `generate_target` writes a PIT snapshot.
  The snapshot includes the new optimization flags (`use_drawdown_control`,
  `use_trend_strength_filter`, `calibration_smooth`, etc.) so Phase 3 replay can
  reconstruct decisions exactly even when research levers are enabled.
- The edge gate ([validate_edge](../scripts/validate_edge.py)) reports panel health
  as context — a PASS on a dirty panel is now visible.
- Tests: `tests/test_phase2_data.py` (15). Full suite 187 green.

## What's next
Phase 3 — replay-based, decision-level drift detection (re-derive each day's target
from its stored snapshot and diff against the recorded decision, separating
logic/config drift from data drift). Now unblocked by the feature store.
