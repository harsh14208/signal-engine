"""Phase 2 — data-quality audit + provider health scorecard.

TRS's provider_reliability scores a live API. A backtest engine's real risk is the
*data panel* itself: silent gaps, staleness, unadjusted-split jumps, and dead
(flatlined) series — the "broken free data, never reconciled" failure the README
was built to avoid. This is the file-based analogue: it scores the loaded panel and
flags symbols before they poison a backtest, plus a lightweight fetch-telemetry log
for the live path.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import market_calendar as cal

_DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_HEALTH_LOG = _DATA_DIR / "provider_health.jsonl"


def audit_symbol(
    prices: pd.Series,
    latest_session: pd.Timestamp | None = None,
    jump_threshold: float = 0.40,
    flatline_run: int = 5,
) -> dict[str, Any]:
    """Quality report + 0–100 health score for one price series.

    Penalties: internal gaps (missing sessions inside the series' own span),
    staleness (last value older than the latest expected session), extreme daily
    jumps (|return| > `jump_threshold` — bad prints or unadjusted corporate
    actions), and flatline runs (≥ `flatline_run` identical closes — a dead feed).
    """
    s = prices.dropna()
    if len(s) < 2:
        return {"insufficient": True, "n": int(len(s)), "health_score": 0.0}

    span_expected = cal.trading_days(s.index.min(), s.index.max())
    internal_gaps = int(len(span_expected.difference(s.index.normalize())))
    gap_frac = internal_gaps / max(len(span_expected), 1)

    latest = latest_session if latest_session is not None else s.index.max()
    stale_sessions = int(len(cal.trading_days(s.index.max(), latest))) - 1
    stale_sessions = max(0, stale_sessions)

    ret = s.pct_change()
    n_jumps = int((ret.abs() > jump_threshold).sum())

    # Longest run of identical consecutive closes.
    flat = (s.diff() == 0).astype(int)
    max_flat = int(flat.groupby((flat == 0).cumsum()).sum().max()) if len(flat) else 0

    score = 100.0
    score -= min(40.0, gap_frac * 400.0)  # 10% gaps → −40
    score -= min(20.0, stale_sessions * 5.0)
    score -= min(25.0, n_jumps * 12.5)
    score -= 15.0 if max_flat >= flatline_run else 0.0
    return {
        "n": int(len(s)),
        "first": s.index.min().strftime("%Y-%m-%d"),
        "last": s.index.max().strftime("%Y-%m-%d"),
        "internal_gaps": internal_gaps,
        "gap_frac": round(gap_frac, 4),
        "stale_sessions": stale_sessions,
        "jumps": n_jumps,
        "max_flatline": max_flat,
        "health_score": round(max(0.0, score), 1),
    }


def audit_panel(
    prices: pd.DataFrame,
    end: str | None = None,
    min_health: float = 70.0,
) -> dict[str, Any]:
    """Panel-wide data-quality scorecard.

    Reports panel-level missing sessions (index vs NYSE calendar), a per-symbol
    audit, the mean health score, and the list of symbols below `min_health` —
    the ones to quarantine before trusting a backtest.
    """
    prices = prices.sort_index()
    if prices.empty:
        return {"insufficient": True}
    latest = pd.Timestamp(end) if end else prices.index.max()
    panel_missing = cal.missing_sessions(prices.index, end=latest)

    per_symbol = {c: audit_symbol(prices[c], latest_session=latest) for c in prices.columns}
    scores = [r["health_score"] for r in per_symbol.values() if "health_score" in r]
    flagged = sorted(
        c for c, r in per_symbol.items() if r.get("health_score", 0.0) < min_health
    )
    return {
        "n_symbols": int(prices.shape[1]),
        "panel_missing_sessions": int(len(panel_missing)),
        "panel_missing_dates": [d.strftime("%Y-%m-%d") for d in panel_missing[:20]],
        "mean_health": round(float(np.mean(scores)), 1) if scores else 0.0,
        "min_health_threshold": min_health,
        "flagged_symbols": flagged,
        "healthy": len(flagged) == 0 and len(panel_missing) == 0,
        "per_symbol": per_symbol,
    }


# ── Live fetch telemetry (append-only, file-based) ───────────────────────────
def record_fetch(
    provider: str,
    ok: bool,
    n_rows: int = 0,
    latency_ms: float = 0.0,
    stale: bool = False,
    note: str = "",
    path: Path | str = DEFAULT_HEALTH_LOG,
) -> None:
    """Append one fetch outcome to the provider-health log (for the live path)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "ok": bool(ok),
        "n_rows": int(n_rows),
        "latency_ms": round(float(latency_ms), 1),
        "stale": bool(stale),
        "note": note,
    }
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def health_scorecard(path: Path | str = DEFAULT_HEALTH_LOG) -> dict[str, Any]:
    """Summarise the fetch log into a per-provider scorecard (TRS-style)."""
    path = Path(path)
    if not path.exists():
        return {"insufficient": True}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return {"insufficient": True}
    df = pd.DataFrame(rows)
    out = {}
    for provider, g in df.groupby("provider"):
        error_rate = float((~g["ok"]).mean())
        stale_rate = float(g["stale"].mean())
        score = 100.0 - error_rate * 60.0 - stale_rate * 25.0
        out[str(provider)] = {
            "n_calls": int(len(g)),
            "error_rate": round(error_rate, 4),
            "stale_rate": round(stale_rate, 4),
            "mean_latency_ms": round(float(g["latency_ms"].mean()), 1),
            "health_score": round(max(0.0, score), 1),
        }
    return out
