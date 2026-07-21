"""Phase 2 — point-in-time feature store (ported from TRS QENG-2a).

Persists an *immutable, content-hashed* snapshot of the exact inputs that produced
each live decision, stamped with the data lineage. Config snapshots already let you
rebuild the model; feature snapshots let you replay the *decision* — the foundation
Phase 3's replay-based drift detection builds on. Immutability is enforced: writing
a different payload for an existing as-of date raises, so history can't be silently
rewritten.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .lineage import lineage_hash

_DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_SNAPSHOT_DIR = _DATA_DIR / "feature_snapshots"


def json_safe(value: Any) -> Any:
    """Recursively replace NaN/Inf with None (JSON can't round-trip them)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def snapshot_hash(features: dict[str, Any]) -> str:
    """Deterministic content hash of a (json-safe) feature payload."""
    blob = json.dumps(json_safe(features), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


FINGERPRINT_VERSION = "v2"


def price_fingerprint(prices) -> str | None:
    """Stable hash of the full price panel — the data identity for an as-of date.

    A change here between snapshot time and replay time means some date's history
    was revised upstream (vendor re-adjustment/backfill), i.e. *data drift*, not
    code drift. Hashing only the latest row (the original implementation) misses
    revisions to earlier bars that the expanding-window calibration is still
    sensitive to — exactly the 2026-07-10 TIP/IEF/HYG/LQD dividend-back-adjustment
    whipsaw documented in data.py's point-in-time cache: the as-of row was
    unchanged, only history behind it moved, so the old fingerprint matched and
    the resulting divergence was misclassified as logic drift.

    The result is prefixed with FINGERPRINT_VERSION so replay.py can tell a
    fingerprint produced by this (full-panel) algorithm apart from one produced
    by the old (last-row-only) algorithm — see fingerprint_comparable().
    """
    if prices is None or len(prices) == 0:
        return None
    rounded = prices.round(6)
    payload = rounded.to_csv()
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"{FINGERPRINT_VERSION}:{digest}"


def fingerprint_comparable(stored_fp: str | None) -> bool:
    """Whether a stored fingerprint was produced by the current price_fingerprint
    algorithm (and so can be equality-compared against a freshly computed one).

    Snapshots saved before FINGERPRINT_VERSION was introduced hold an unversioned
    hash from the old (last-row-only) algorithm; comparing those against a new
    full-panel hash would always mismatch regardless of whether the data actually
    changed, which would misclassify real logic drift as benign data drift.
    """
    return stored_fp is not None and stored_fp.startswith(f"{FINGERPRINT_VERSION}:")


def save_snapshot(
    as_of: str,
    features: dict[str, Any],
    book: str = "champion",
    snapshot_dir: Path | str = DEFAULT_SNAPSHOT_DIR,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Persist an immutable feature snapshot keyed by (as_of, book).

    Returns the stored envelope. Re-saving identical content is a no-op; saving
    *different* content for an existing key raises unless `overwrite=True`.
    """
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    safe = json_safe(features)
    envelope = {
        "as_of": as_of,
        "book": book,
        "content_hash": snapshot_hash(safe),
        "lineage_hash": lineage_hash(),
        "features": safe,
    }
    path = snapshot_dir / f"{as_of}_{book}.json"
    if path.exists() and not overwrite:
        existing = json.loads(path.read_text())
        if existing.get("content_hash") != envelope["content_hash"]:
            raise ValueError(
                f"immutable snapshot conflict for {as_of}/{book}: "
                f"{existing.get('content_hash')} != {envelope['content_hash']}"
            )
        return existing
    path.write_text(json.dumps(envelope, indent=2))
    return envelope


def load_snapshot(
    as_of: str, book: str = "champion", snapshot_dir: Path | str = DEFAULT_SNAPSHOT_DIR
) -> dict[str, Any] | None:
    """Load a stored snapshot, or None if it doesn't exist."""
    path = Path(snapshot_dir) / f"{as_of}_{book}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def list_snapshots(snapshot_dir: Path | str = DEFAULT_SNAPSHOT_DIR) -> list[str]:
    """Sorted as-of keys of all stored snapshots."""
    d = Path(snapshot_dir)
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


def snapshot_from_target(record: dict[str, Any], prices) -> dict[str, Any]:
    """Build the decision-input feature payload for a target record.

    Captures the causal inputs that produced the target: the forecast/units, the
    calibrated scalars, a fingerprint of the price panel used, and COT as-of — so
    the decision can be replayed and diffed against a re-computation (Phase 3).
    """
    return {
        "date": record.get("date"),
        "book": record.get("book", "champion"),
        "config": {
            "capital": record.get("capital"),
            "vol_target": record.get("vol_target"),
            "buffer_fraction": record.get("buffer_fraction"),
            "use_cot": record.get("use_cot"),
            "cot_momentum": record.get("cot_momentum"),
        },
        "idm": record.get("idm"),
        "fdm": record.get("fdm"),
        "governor": record.get("governor"),
        "cot_as_of": record.get("cot_as_of"),
        "forecast": record.get("forecast"),
        "units": record.get("units"),
        "price_panel_asof": record.get("as_of"),
        "price_fingerprint": price_fingerprint(prices),
    }
