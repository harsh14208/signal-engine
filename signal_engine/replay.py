"""Phase 3 — replay-based, decision-level drift detection (ported from TRS REF-2).

Return-level reconciliation (`monitor.reconcile`) tells you *that* live diverged
from the model. This tells you *why*, at the decision level: it re-derives each
recorded target from its point-in-time feature snapshot and diffs the re-computed
units/forecast against what was actually stored, then classifies the disagreement:

  • **logic**   — same data, lineage, and config, but the engine now produces a
                  different decision ⇒ the CODE changed (a bug or silent logic
                  drift). This is the alarming case.
  • **data**    — the price fingerprint changed ⇒ the vendor revised history under
                  that as-of date (re-adjustment/backfill). Not a code bug.
  • **lineage** — the data-handling regime (`lineage_hash`) changed.
  • **matched** — the decision reproduces exactly. Healthy.

Config/policy changes are reported separately (`policy_changed`): replaying with
the snapshot's own config means an intentional default change never masquerades as
logic drift — exactly REF-2's policy-vs-logic split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .backtest import run_backtest
from .config import Config
from .feature_store import DEFAULT_SNAPSHOT_DIR, fingerprint_comparable, load_snapshot, price_fingerprint
from .lineage import lineage_hash

_CONFIG_FIELDS = ("capital", "vol_target", "buffer_fraction", "use_cot", "cot_momentum")


def config_from_snapshot(features: dict[str, Any]) -> Config:
    """Rebuild the exact Config that produced a snapshot's decision."""
    from .live import validated_config

    cfg = features.get("config", {})
    return validated_config(
        cot=cfg.get("use_cot", True),
        capital=cfg.get("capital", 1_000_000.0),
        vol_target=cfg.get("vol_target", 0.20),
        buffer_fraction=cfg.get("buffer_fraction", 0.30),
        cot_momentum=cfg.get("cot_momentum", False),
    )


def _max_abs_diff(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """(max abs difference, scale) over the union of two label→value maps."""
    idx = a.index.union(b.index)
    if len(idx) == 0:
        return 0.0, 1.0
    aa = a.reindex(idx).fillna(0.0)
    bb = b.reindex(idx).fillna(0.0)
    scale = max(1.0, float(pd.concat([aa.abs(), bb.abs()]).max()))
    return float((aa - bb).abs().max()), scale


def replay_decision(
    snapshot: dict[str, Any],
    prices: pd.DataFrame | None = None,
    source: str = "cache",
    cot: pd.DataFrame | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-4,
) -> dict[str, Any]:
    """Re-derive one snapshot's decision and classify any drift.

    Pass `prices` (and `cot`) directly for offline/testing; otherwise the price
    panel up to the snapshot's as-of date is reloaded from `source`.
    """
    feats = snapshot.get("features", snapshot)
    as_of = feats.get("price_panel_asof") or feats.get("date")
    cfg = config_from_snapshot(feats)

    if prices is None:
        from .cot_data import build_cot_forecast_panel
        from .data import load_prices
        from .markets import symbols

        prices = load_prices(symbols(), start="2007-01-01", end=as_of,
                             source=source, cache_tag="universe")
        prices = prices.loc[prices.index <= pd.Timestamp(as_of)]
        if cot is None and cfg.use_cot:
            cot = build_cot_forecast_panel(prices, tag="core", momentum=cfg.cot_momentum)

    result = run_backtest(prices, cfg, cot=cot)
    replay_units = result.buffered.iloc[-1]
    replay_forecast = result.forecasts.iloc[-1]
    replay_fp = price_fingerprint(prices)

    stored_units = pd.Series(feats.get("units") or {}, dtype=float)
    stored_forecast = pd.Series(feats.get("forecast") or {}, dtype=float)
    stored_fp = feats.get("price_fingerprint")

    u_diff, u_scale = _max_abs_diff(replay_units, stored_units)
    f_diff, f_scale = _max_abs_diff(replay_forecast, stored_forecast)
    units_ok = u_diff <= atol + rtol * u_scale
    forecast_ok = f_diff <= atol + rtol * f_scale

    data_drift = fingerprint_comparable(stored_fp) and replay_fp != stored_fp
    lineage_drift = snapshot.get("lineage_hash") is not None and (
        snapshot["lineage_hash"] != lineage_hash()
    )
    policy_changed = _policy_changed(feats)

    if units_ok and forecast_ok:
        kind, alarm = "matched", False
    elif data_drift:
        kind, alarm = "data", False
    elif lineage_drift:
        kind, alarm = "lineage", False
    else:
        kind, alarm = "logic", True  # same inputs, different output ⇒ code drift

    return {
        "as_of": as_of,
        "book": feats.get("book", "champion"),
        "kind": kind,
        "alarm": alarm,
        "units_max_diff": u_diff,
        "forecast_max_diff": f_diff,
        "data_drift": data_drift,
        "lineage_drift": lineage_drift,
        "policy_changed": policy_changed,
    }


def _policy_changed(features: dict[str, Any]) -> bool:
    """True if today's default config differs from the snapshot's config."""
    from .live import validated_config

    snap_cfg = features.get("config", {})
    current = validated_config(cot=snap_cfg.get("use_cot", True))
    for f in _CONFIG_FIELDS:
        if f in snap_cfg and getattr(current, f, None) != snap_cfg[f]:
            return True
    return False


def detect_drift(
    snapshot_dir: Path | str = DEFAULT_SNAPSHOT_DIR,
    source: str = "cache",
    prices: pd.DataFrame | None = None,
    cot: pd.DataFrame | None = None,
    books: list[str] | None = None,
) -> dict[str, Any]:
    """Replay every stored snapshot and aggregate a drift report.

    `alarm` is set iff any snapshot shows *logic* drift (a code-level divergence).
    Data/lineage/policy drift are counted but do not alarm.
    """
    snapshot_dir = Path(snapshot_dir)
    keys = sorted(p.stem for p in snapshot_dir.glob("*.json")) if snapshot_dir.exists() else []
    results, unverified = [], []
    by_kind: dict[str, int] = {}

    for key in keys:
        # key = "<as_of>_<book>"
        as_of, _, book = key.rpartition("_")
        if books and book not in books:
            continue
        snap = load_snapshot(as_of, book=book, snapshot_dir=snapshot_dir)
        if snap is None:
            continue
        try:
            res = replay_decision(snap, prices=prices, source=source, cot=cot)
        except Exception as exc:
            unverified.append({"as_of": as_of, "book": book, "error": str(exc)})
            continue
        results.append(res)
        by_kind[res["kind"]] = by_kind.get(res["kind"], 0) + 1

    alarms = [r for r in results if r["alarm"]]
    return {
        "n_snapshots": len(results),
        "matched": by_kind.get("matched", 0),
        "by_kind": by_kind,
        "policy_changes": sum(1 for r in results if r["policy_changed"]),
        "unverified": unverified,
        "alarms": alarms,
        "alarm": len(alarms) > 0,
    }
