# Phase 3 — replay-based decision-level drift detection

Turns return-level reconciliation into *decision-level* verification. Ported from
TRS's REF-2, built on the Phase 2 point-in-time feature store.

## What it answers
`monitor.reconcile` says live returns diverged from the model. This says **why**,
by re-deriving each recorded target from its stored snapshot and diffing the
re-computed decision against what was actually stored:

| Drift kind | Cause | Alarms? |
|---|---|---|
| **matched** | decision reproduces exactly | — |
| **logic** | same data + lineage + config, different output ⇒ **code changed** | ✅ yes |
| **data** | price fingerprint changed ⇒ vendor revised history under that date | no |
| **lineage** | data-handling regime (`lineage_hash`) changed | no |
| **policy_changed** | today's default config ≠ the snapshot's config (intentional) | no (reported) |

Only **logic drift** alarms — it's the code-level divergence a bug would cause.
Replaying with the snapshot's *own* config is what keeps an intentional default
change from masquerading as a bug (REF-2's policy-vs-logic split).

## Usage
```bash
python scripts/detect_drift.py --source cache            # report only
python scripts/detect_drift.py --source cache --enforce  # + kill switch on logic drift
```
Exit 0 = clean, 1 = logic-drift alarm, 3 = error. Now runs as step 5 of
`scripts/forward_loop.sh` (with `--enforce`).

Library:
```python
from signal_engine.replay import replay_decision, detect_drift, config_from_snapshot
replay_decision(snapshot, prices=None, source="cache")   # one snapshot
detect_drift(snapshot_dir=..., source="cache")           # aggregate + alarm
```

## How replay works
1. Rebuild the exact Config from the snapshot (`config_from_snapshot`).
2. Reload prices up to the snapshot's as-of date (or inject a panel for tests).
3. Re-run the backtest; take the latest-row units/forecast.
4. Diff vs the stored units/forecast (scale-aware tolerance) and compare the price
   fingerprint + lineage hash → classify.

## Verified behaviour (`tests/test_phase3_replay.py`, 8 tests)
- Faithful replay → `matched`, Δ0.0.
- Tampered units (code-drift proxy) → `logic`, **alarm**.
- Tampered units + fingerprint (revised data) → `data`, no alarm.
- Tampered units + lineage → `lineage`, no alarm.
- `detect_drift` aggregates and trips the alarm only on logic drift.

Full suite: 195 tests green, lint clean.

## Phases 0–3 recap
| Phase | Capability | Entry point |
|---|---|---|
| 0 | edge go/no-go gate | `scripts/validate_edge.py` |
| 1 | realized-friction / cost break-even | `signal_engine/friction.py` |
| 2 | calendar + data quality + PIT feature store + lineage | `scripts/audit_data.py` |
| 3 | replay-based decision drift detection | `scripts/detect_drift.py` |

Remaining (Phase 4, optional): HRP allocator + PCA risk model — justified only if
the edge holds on real-futures data and a live track accrues.
