"""Phase 2 — data lineage / provenance (ported from TRS QENG-2c).

A single versioned descriptor of *where the data came from and how it was treated*,
hashed so every live decision and feature snapshot can be stamped with the exact
data-handling regime that produced it. When live diverges from backtest, the first
question is "did the data lineage change?" — this makes that answerable.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from .markets import symbols

DATA_LINEAGE_VERSION = "v1.0.0"

LINEAGE_CONFIG: dict = {
    "data_version": DATA_LINEAGE_VERSION,
    "vendor": "yfinance",
    "price_field": "adjusted_close",
    "adjustment_mode": "split_and_dividend_adjusted",  # yfinance auto_adjust=True
    "market_calendar": "NYSE",
    "fill_policy": "forward_fill",
    "min_history_bars": 300,
    "schema_versions": {
        "target_record": "v2.0",  # added `book`
        "live_returns": "v2.0",  # added `book`, `delay_return`
        "feature_snapshot": "v1.0",
        "trial_registry": "v1.0",
    },
}


def universe_hash(expanded: bool = False) -> str:
    """Stable hash of the instrument universe (symbol set)."""
    blob = json.dumps(sorted(symbols(expanded)), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def lineage_hash() -> str:
    """Content hash of the lineage config + core universe (provenance fingerprint)."""
    blob = json.dumps({**LINEAGE_CONFIG, "universe": universe_hash()}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def lineage_meta() -> dict:
    """Full provenance block to stamp into records/snapshots."""
    return {
        "lineage_hash": lineage_hash(),
        "universe_hash": universe_hash(),
        "stamped_at": datetime.now(timezone.utc).isoformat(),
        **LINEAGE_CONFIG,
    }
