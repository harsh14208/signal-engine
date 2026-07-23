"""One-time migration: backfill legacy (unversioned) price_fingerprint values.

Context (2026-07-22 incident): price_fingerprint gained a FINGERPRINT_VERSION
prefix ("v2:") so replay.py could tell a fingerprint from the new full-panel
algorithm apart from one produced by the old last-row-only algorithm — mixing
the two would either mask real logic drift as benign data drift, or (as
happened here) the reverse: every snapshot saved before the version prefix
existed is now permanently "not comparable", so a genuine (already known,
benign) vendor price revision on one of those dates falls through to LOGIC
drift instead of DATA drift and trips the kill switch.

This script re-stamps every legacy snapshot's price_fingerprint using the
CURRENT point-in-time price panel (truncated to that snapshot's as-of date),
so that as of today those snapshots compare cleanly against future replays —
any *further* revision from today onward will still be caught correctly, but
revisions that already happened before today (TIP/IEF/LQD/HYG dividend
back-adjustments — already independently flagged every day by reconcile.py's
input_revision check) are absorbed into the new baseline rather than
re-litigated. content_hash is recomputed alongside so the immutability guard
in save_snapshot() doesn't fire if one of these dates is ever regenerated.

Usage: .venv/bin/python scripts/backfill_price_fingerprint_v2.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from signal_engine.data import load_prices  # noqa: E402
from signal_engine.feature_store import (  # noqa: E402
    DEFAULT_SNAPSHOT_DIR,
    fingerprint_comparable,
    json_safe,
    price_fingerprint,
    snapshot_hash,
)
from signal_engine.markets import symbols  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOT_DIR))
    args = parser.parse_args(argv)

    snapshot_dir = Path(args.snapshot_dir)
    paths = sorted(snapshot_dir.glob("*.json"))
    if not paths:
        print("No snapshots found.")
        return 0

    as_ofs = []
    for p in paths:
        try:
            envelope = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        feats = envelope.get("features", envelope)
        as_of = feats.get("price_panel_asof") or feats.get("date") or envelope.get("as_of")
        if as_of:
            as_ofs.append(as_of)
    if not as_ofs:
        print("No as-of dates found in any snapshot.")
        return 0

    prices_full = load_prices(symbols(), start="2007-01-01", end=max(as_ofs), source="cache", cache_tag="universe")

    n_updated = 0
    for p in paths:
        envelope = json.loads(p.read_text())
        feats = envelope.get("features", envelope)
        stored_fp = feats.get("price_fingerprint")
        if fingerprint_comparable(stored_fp):
            continue  # already v2 — nothing to do
        as_of = feats.get("price_panel_asof") or feats.get("date") or envelope.get("as_of")
        if not as_of:
            print(f"SKIP {p.name}: no as-of date found")
            continue
        prices = prices_full.loc[prices_full.index <= pd.Timestamp(as_of)]
        new_fp = price_fingerprint(prices)
        print(f"{p.name}: {stored_fp} -> {new_fp}")
        n_updated += 1
        if args.dry_run:
            continue
        feats["price_fingerprint"] = new_fp
        envelope["content_hash"] = snapshot_hash(json_safe(feats))
        p.write_text(json.dumps(envelope, indent=2))

    print(f"{'Would update' if args.dry_run else 'Updated'} {n_updated} snapshot(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
