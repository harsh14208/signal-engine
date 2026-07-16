"""Evaluate Phase 1 free breadth levers: crypto pack and curated expanded universe.

All runs use a realistic 3× gross cap + 1% financing so results are comparable to
the earlier options evaluation. Walk-forward OOS Sharpe is the honest bar, with
paired fold comparison vs baseline.
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
from signal_engine.edge_gate import promotion_decision
from signal_engine.markets import symbols
from signal_engine.metrics import summary
from signal_engine.validation import paired_fold_comparison, purged_walk_forward


CORE = symbols(expanded=False)
END = "2026-07-15"
CACHE_TAG = "breadth_experiment"
N_SPLITS = 4
CAP = 3.0
FIN_RATE = 0.01


def _run(label: str, syms: list[str], cfg: Config) -> dict:
    prices = load_prices(
        syms,
        start="2007-01-01",
        end=END,
        source="yfinance",
        cache_tag=CACHE_TAG,
        rebase=True,
    )
    panel = prices[syms].dropna(how="all").copy()
    carry = build_carry_panel(panel, cfg) if cfg.use_carry or cfg.use_carry_proxies else None
    cot = build_cot_forecast_panel(panel, cfg) if cfg.use_cot else None
    result = run_backtest(panel, cfg, carry=carry, cot=cot)
    s = summary(result.equity, result.daily_returns, result.turnover)
    wf = purged_walk_forward(panel, cfg, n_splits=N_SPLITS, embargo_frac=0.02)
    return {
        "label": label,
        "symbols": syms,
        "net_sharpe": s["sharpe"],
        "ann_vol": s["ann_vol"],
        "max_dd": s["max_drawdown"],
        "turnover": s["ann_turnover"],
        "mean_gross": float(result.gross_exposure.mean()),
        "wf_mean_is": wf.get("mean_is_sharpe"),
        "wf_mean_oos": wf.get("mean_oos_sharpe"),
        "wf_gap": wf.get("mean_gap"),
        "wf_folds": wf.get("folds", []),
    }


def main() -> None:
    base_cfg = Config(max_gross_notional=CAP, financing_rate=FIN_RATE)

    scenarios = [
        ("baseline", CORE, base_cfg),
        ("+crypto", CORE + ["BTC-USD", "ETH-USD"], Config(**{**base_cfg.__dict__, "use_crypto": True})),
        (
            "+curated breadth",
            CORE + ["UNG", "CPER", "CORN", "WEAT", "SOYB", "EMB", "BWX", "FXA", "FXB", "FXC"],
            Config(**{**base_cfg.__dict__, "use_curated_breadth": True}),
        ),
        (
            "+crypto +curated breadth",
            CORE
            + ["BTC-USD", "ETH-USD"]
            + ["UNG", "CPER", "CORN", "WEAT", "SOYB", "EMB", "BWX", "FXA", "FXB", "FXC"],
            Config(**{**base_cfg.__dict__, "use_crypto": True, "use_curated_breadth": True}),
        ),
        (
            "+semis",
            CORE + ["SMH", "SOXX", "XSD"],
            base_cfg,
        ),
        (
            "+crypto +curated breadth +semis",
            CORE
            + ["BTC-USD", "ETH-USD"]
            + ["UNG", "CPER", "CORN", "WEAT", "SOYB", "EMB", "BWX", "FXA", "FXB", "FXC"]
            + ["SMH", "SOXX", "XSD"],
            Config(**{**base_cfg.__dict__, "use_crypto": True, "use_curated_breadth": True}),
        ),
    ]

    rows = []
    for label, syms, cfg in scenarios:
        print(f"Running {label}...")
        rows.append(_run(label, syms, cfg))

    baseline = rows[0]

    print("\n" + "=" * 110)
    print(
        f"{'Configuration':<36} {'Net SR':>8} {'WF OOS':>8} {'Gap':>8} "
        f"{'Max DD':>9} {'Turn':>8} {'Gross':>8}"
    )
    print("-" * 110)
    for r in rows:
        print(
            f"{r['label']:<36} {r['net_sharpe']:>8.3f} {r['wf_mean_oos']:>8.3f} "
            f"{r['wf_gap']:>8.3f} {r['max_dd']:>9.1%} {r['turnover']:>8.1f}x {r['mean_gross']:>8.2f}x"
        )

    print("\nPaired fold comparison vs baseline (bootstrap 95% CI on OOS Sharpe delta):")
    print("-" * 110)
    for r in rows[1:]:
        comp = paired_fold_comparison(baseline["wf_folds"], r["wf_folds"])
        if comp.get("insufficient"):
            print(f"{r['label']:<36} insufficient common folds")
            continue
        ci = f"[{comp['ci_low']:+.3f}, {comp['ci_high']:+.3f}]"
        tag = "indistinguishable" if comp["indistinguishable"] else "significant"
        promo = promotion_decision(baseline, r)
        print(
            f"{r['label']:<36} delta={comp['mean_delta']:+.3f} {ci:<18} {tag:<17} "
            f"promotion={promo['verdict']}"
        )

    out = "data/breadth_levers_evaluation.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
