"""Experiment registry for honest Deflated-Sharpe trial counting.

The hardcoded `n_trials = 100` is a placeholder; as the project grows the
Deflated bar should reflect the actual number of distinct configurations
searched. This module appends every run to a JSONL log and counts unique
config hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone

from .backtest import BacktestResult
from .config import Config

EXPERIMENT_LOG = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "experiments.jsonl"
)


def _config_hash(config: Config) -> str:
    """Stable hash of a Config so duplicate runs are not double-counted."""
    # asdict returns nested tuples; json cannot serialise tuples directly.
    payload = json.dumps(asdict(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def log_experiment(
    config: Config,
    result: BacktestResult,
    path: str | None = None,
) -> None:
    """Append one experiment record to the JSONL log."""
    path = path or EXPERIMENT_LOG
    os.makedirs(os.path.dirname(path), exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": _config_hash(config),
        "config": asdict(config),
        "sharpe": float(result.daily_returns.mean() / result.daily_returns.std() * 16)
        if result.daily_returns.std() > 0
        else 0.0,
        "idm": float(result.idm),
        "fdm": float(result.fdm),
        "n_instruments": int(result.per_instrument_returns.shape[1]),
        "n_days": int(result.daily_returns.shape[0]),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def count_experiments(path: str | None = None) -> int:
    """Return the number of unique config hashes in the log."""
    path = path or EXPERIMENT_LOG
    if not os.path.exists(path):
        return 0
    seen: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            h = record.get("config_hash")
            if h:
                seen.add(h)
    return len(seen)
