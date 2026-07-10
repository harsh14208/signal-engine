"""Phase 1 — realized-friction calibration (ported method from TRS §104b).

The engine's entire *net* Sharpe rests on one assumed number: 1.5 bps per side.
This module makes that number empirical and testable, three ways:

  1. `realized_friction`  — back out actual per-side slippage + commission from
     broker fills (decision price vs fill price), per instrument and overall.
     Mirrors TRS's rule: *measure* the real cost, don't silently change the
     assumption.
  2. `net_of_friction_curve` / `cost_break_even` — re-run the backtest across a
     grid of cost assumptions and find the round-trip cost at which the edge
     dies. Runnable today; the honest robustness test of the 1.5 bps figure.
  3. `write_calibration` / `load_calibration` — persist measured per-symbol costs
     so `Config(cost_scheme="calibrated")` prices the backtest off real fills.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import Config
from .metrics import sharpe

_DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_CALIBRATION_PATH = _DATA_DIR / "friction_calibration.json"


def slippage_bps(decision_price: float, fill_price: float, side: str) -> float:
    """Signed execution slippage in bps: positive = worse than the decision price.

    A buy that fills *above* the decision price costs you (positive); a sell that
    fills *below* costs you (positive). This is the arrival-vs-fill implementation
    shortfall at the single-fill level.
    """
    if decision_price <= 0:
        return 0.0
    raw = (fill_price - decision_price) / decision_price * 1e4
    return raw if side.lower() == "buy" else -raw


def realized_friction(fills: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Realized transaction cost from broker fills.

    Each fill dict needs: ``symbol``, ``side``, ``decision_price``, ``fill_price``,
    and optionally ``qty`` and ``commission`` (absolute currency) with ``notional``.
    Returns overall + per-symbol slippage stats (mean/median/p90 bps), mean
    commission bps, and an estimated round-trip cost (2 × per-side slippage +
    commission) — the empirical analogue of the assumed per-side ``cost_bps``.
    """
    rows = []
    for f in fills:
        dp = float(f.get("decision_price") or 0.0)
        fp = float(f.get("fill_price") or 0.0)
        if dp <= 0 or fp <= 0:
            continue
        slip = slippage_bps(dp, fp, str(f.get("side", "buy")))
        notional = float(f.get("notional") or (float(f.get("qty") or 0.0) * fp))
        comm = float(f.get("commission") or 0.0)
        comm_bps = (comm / notional * 1e4) if notional > 0 else 0.0
        rows.append({"symbol": str(f.get("symbol", "?")), "slippage_bps": slip, "commission_bps": comm_bps})

    if not rows:
        return {"insufficient": True, "n_fills": 0}

    df = pd.DataFrame(rows)

    def _stats(g: pd.DataFrame) -> dict[str, float]:
        s = g["slippage_bps"]
        per_side = float(s.mean()) + float(g["commission_bps"].mean())
        return {
            "n_fills": int(len(g)),
            "mean_slippage_bps": float(s.mean()),
            "median_slippage_bps": float(s.median()),
            "p90_slippage_bps": float(s.quantile(0.90)),
            "mean_commission_bps": float(g["commission_bps"].mean()),
            "per_side_bps": per_side,
            "round_trip_bps": 2.0 * per_side,
        }

    overall = _stats(df)
    overall["n_symbols"] = int(df["symbol"].nunique())
    overall["per_symbol"] = {sym: _stats(g) for sym, g in df.groupby("symbol")}
    return overall


def net_of_friction_curve(
    prices: pd.DataFrame,
    config: Config | None = None,
    costs_bps: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0),
    cot: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Net Sharpe as a function of the assumed per-side cost.

    Re-runs the backtest at each cost so you can see how much margin the edge has
    over the 1.5 bps assumption — the difference between a robust edge and one
    that only exists because costs were assumed too low.
    """
    from .backtest import run_backtest

    base = config or Config()
    rows = []
    for cost in costs_bps:
        cfg = _with_cost(base, cost)
        res = run_backtest(prices, cfg, cot=cot)
        rows.append(
            {
                "cost_bps": cost,
                "net_sharpe": sharpe(res.daily_returns),
                "gross_sharpe": sharpe(res.gross_returns),
                "ann_turnover": float(res.turnover.mean() * 256),
            }
        )
    return pd.DataFrame(rows)


def cost_break_even(
    prices: pd.DataFrame,
    config: Config | None = None,
    cot: pd.DataFrame | None = None,
    max_cost_bps: float = 25.0,
) -> dict[str, Any]:
    """Per-side cost at which net Sharpe crosses zero (the edge's cost headroom).

    Found by linear interpolation over a fine cost grid. ``headroom_x`` reports
    the break-even as a multiple of the assumed cost — a headroom below ~2× means
    the net edge is dangerously sensitive to the cost assumption.
    """
    grid = tuple(np.round(np.arange(0.0, max_cost_bps + 0.5, 0.5), 3))
    curve = net_of_friction_curve(prices, config, costs_bps=grid, cot=cot)
    assumed = (config or Config()).cost_bps

    below = curve[curve["net_sharpe"] <= 0]
    if below.empty:
        return {
            "break_even_bps": None,
            "note": f"net Sharpe stays positive through {max_cost_bps} bps",
            "assumed_cost_bps": assumed,
            "curve": curve.to_dict("records"),
        }
    # Interpolate between the last positive and first non-positive point.
    first_neg_idx = below.index[0]
    if first_neg_idx == 0:
        be = 0.0
    else:
        x0, y0 = curve.loc[first_neg_idx - 1, ["cost_bps", "net_sharpe"]]
        x1, y1 = curve.loc[first_neg_idx, ["cost_bps", "net_sharpe"]]
        be = float(x0 + (x1 - x0) * y0 / (y0 - y1)) if y0 != y1 else float(x0)
    return {
        "break_even_bps": be,
        "assumed_cost_bps": assumed,
        "headroom_x": (be / assumed) if assumed > 0 else None,
        "curve": curve.to_dict("records"),
    }


def _with_cost(config: Config, cost_bps: float) -> Config:
    """Clone a Config with a flat per-side cost (used by the sensitivity sweep)."""
    from dataclasses import replace

    return replace(config, cost_bps=float(cost_bps), cost_scheme="flat")


# ── Calibration persistence ──────────────────────────────────────────────────
def write_calibration(
    friction: dict[str, Any], path: Path | str = DEFAULT_CALIBRATION_PATH
) -> dict[str, float]:
    """Persist measured per-symbol per-side costs for `cost_scheme="calibrated"`.

    Uses each symbol's per-side bps (falling back to the overall figure). Writes a
    flat ``symbol -> bps`` map plus provenance so the calibration is auditable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if friction.get("insufficient"):
        raise ValueError("cannot calibrate from insufficient fills")
    per_symbol = {
        sym: round(float(stats["per_side_bps"]), 3)
        for sym, stats in friction.get("per_symbol", {}).items()
    }
    payload = {
        "per_side_bps": per_symbol,
        "default_bps": round(float(friction["per_side_bps"]), 3),
        "n_fills": int(friction.get("n_fills", 0)),
    }
    path.write_text(json.dumps(payload, indent=2))
    return per_symbol


def load_calibration(path: Path | str = DEFAULT_CALIBRATION_PATH) -> dict[str, Any]:
    """Load the calibrated per-side cost map (empty dict if none exists)."""
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())
