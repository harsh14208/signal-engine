"""Validate free orthogonal rules: acceleration, cross-sectional momentum, breakout spans.

Runs baseline, +accel, +xsmom, +both, and a longer-breakout configuration through
purged walk-forward on the cached core universe. Saves results to JSON and prints
a table. None of these rules has improved the walk-forward OOS Sharpe so far, so
they remain RESEARCH flags.
"""

# ruff: noqa: E402

from __future__ import annotations

import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings(
    "ignore",
    message=".*--max-gross=.* with --financing-rate=0 overstates levered strategies.*",
    category=UserWarning,
)

from signal_engine.backtest import run_backtest
from signal_engine.carry_data import build_carry_panel
from signal_engine.config import Config
from signal_engine.cot_data import build_cot_forecast_panel
from signal_engine.data import load_prices
from signal_engine.markets import symbols
from signal_engine.metrics import summary
from signal_engine.validation import paired_fold_comparison, purged_walk_forward


CORE = symbols(expanded=False)
END = "2026-07-15"
CACHE_TAG = "universe"
N_SPLITS = 4


def _run(label: str, cfg: Config) -> dict:
    prices = load_prices(CORE, start="2007-01-01", end=END, source="cache", cache_tag=CACHE_TAG)
    panel = prices[CORE].dropna(how="all").copy()
    carry = build_carry_panel(panel, cfg) if cfg.use_carry or cfg.use_carry_proxies else None
    cot = build_cot_forecast_panel(panel, cfg) if cfg.use_cot else None
    result = run_backtest(panel, cfg, carry=carry, cot=cot)
    s = summary(result.equity, result.daily_returns, result.turnover)
    wf = purged_walk_forward(panel, cfg, n_splits=N_SPLITS, embargo_frac=0.02)
    return {
        "label": label,
        "net_sharpe": s["sharpe"],
        "ann_vol": s["ann_vol"],
        "max_dd": s["max_drawdown"],
        "turnover": s["ann_turnover"],
        "wf_mean_is": wf.get("mean_is_sharpe"),
        "wf_mean_oos": wf.get("mean_oos_sharpe"),
        "wf_gap": wf.get("mean_gap"),
        "wf_folds": wf.get("folds", []),
    }


def main() -> None:
    configs = [
        ("baseline", Config()),
        ("+accel", Config(use_accel=True)),
        ("+xsmom", Config(use_xsmom=True)),
        ("+accel +xsmom", Config(use_accel=True, use_xsmom=True)),
        ("longer breakout (80,160,320)", Config(breakout_spans=(80, 160, 320))),
        ("+all three", Config(use_accel=True, use_xsmom=True, breakout_spans=(80, 160, 320))),
    ]

    rows = []
    for label, cfg in configs:
        print(f"Running {label}...")
        rows.append(_run(label, cfg))

    baseline = rows[0]
    print("\n" + "=" * 95)
    print(f"{'Configuration':<32} {'Net SR':>8} {'WF OOS':>8} {'Gap':>8} {'Max DD':>9} {'Turn':>8}")
    print("-" * 95)
    for r in rows:
        print(
            f"{r['label']:<32} {r['net_sharpe']:>8.3f} {r['wf_mean_oos']:>8.3f} "
            f"{r['wf_gap']:>8.3f} {r['max_dd']:>9.1%} {r['turnover']:>8.1f}x"
        )

    print("\nPaired fold comparison vs baseline (bootstrap 95% CI on OOS Sharpe delta):")
    print("-" * 95)
    for r in rows[1:]:
        comp = paired_fold_comparison(baseline["wf_folds"], r["wf_folds"])
        if comp.get("insufficient"):
            print(f"{r['label']:<32} insufficient common folds")
            continue
        ci = f"[{comp['ci_low']:+.3f}, {comp['ci_high']:+.3f}]"
        tag = "indistinguishable" if comp["indistinguishable"] else "significant"
        print(
            f"{r['label']:<32} delta={comp['mean_delta']:+.3f} {ci:<18} {tag}"
        )

    out = "data/free_rules_evaluation.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
