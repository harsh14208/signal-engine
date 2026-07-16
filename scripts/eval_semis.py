"""Evaluate adding semiconductor ETFs (SMH, SOXX, XSD) to the core universe.

Run from repo root:
    source .venv/bin/activate && python scripts/eval_semis.py

Fetches/uses yfinance data for the core 19 plus the three semi ETFs, then
compares the default config on:
  - core 19 (baseline)
  - core + SMH
  - core + SOXX
  - core + XSD
  - core + all three semis

Reports full-sample metrics, walk-forward OOS (the honest test), and
correlations of the new ETFs to existing instruments.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Allow running from repo root without install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_engine.backtest import run_backtest
from signal_engine.config import Config
from signal_engine.data import load_prices
from signal_engine.experiments import log_experiment
from signal_engine.markets import symbols as core_symbols
from signal_engine.metrics import sharpe, summary
from signal_engine.report import full_report
from signal_engine.validation import purged_walk_forward


SEMI_ETFS = ["SMH", "SOXX", "XSD"]
CACHE_TAG = "semis_experiment"
START = "2007-01-01"
# Use a fixed end date so all variants share the exact same calendar.
END = "2026-07-14"
N_WF_SPLITS = 5  # purged_walk_forward(n_splits=5) yields 4 folds


def _fmt(x: float | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.2f}"


def _run_variant(name: str, syms: list[str], prices: pd.DataFrame) -> dict:
    """Run a full-sample backtest + walk-forward for one symbol list."""
    panel = prices[syms].dropna(how="all")
    cfg = Config()  # default validated config: governor ON, equal weights, trend-only
    print(f"\n=== {name} ({len(syms)} instruments, {len(panel)} days) ===")

    result = run_backtest(panel, cfg)
    print(full_report(result))
    log_experiment(cfg, result)

    print(f"\n### Walk-forward ({N_WF_SPLITS}-fold)")
    wf = purged_walk_forward(panel, cfg, n_splits=N_WF_SPLITS)
    if wf.get("insufficient"):
        print("- Insufficient data for walk-forward.")
    else:
        print(f"- mean IS Sharpe: {wf['mean_is_sharpe']:.2f}")
        print(f"- mean OOS Sharpe: {wf['mean_oos_sharpe']:.2f}")
        print(f"- mean gap (IS − OOS): {wf['mean_gap']:+.2f}")
        for f in wf["folds"]:
            print(
                f"  {f['test_start'][:10]} → {f['test_end'][:10]}: "
                f"IS {f['is_sharpe']:.2f} / OOS {f['oos_sharpe']:.2f}"
            )

    s = summary(result.equity, result.daily_returns, result.turnover)
    return {
        "name": name,
        "n_instruments": len(syms),
        "net_sharpe": s["sharpe"],
        "gross_sharpe": sharpe(result.gross_returns),
        "ann_return": s["ann_return"],
        "ann_vol": s["ann_vol"],
        "max_dd": s["max_drawdown"],
        "calmar": s["calmar"],
        "idm": result.idm,
        "fdm": result.fdm,
        "mean_standalone": float(np.nanmean([sharpe(result.per_instrument_returns[c]) for c in result.per_instrument_returns.columns])),
        "mean_pairwise_corr": float(_avg_offdiag_corr(result.per_instrument_returns)),
        "diversification_ratio": s["sharpe"] / float(np.nanmean([sharpe(result.per_instrument_returns[c]) for c in result.per_instrument_returns.columns])) if float(np.nanmean([sharpe(result.per_instrument_returns[c]) for c in result.per_instrument_returns.columns])) not in (0, np.nan) else float("nan"),
        "wf_mean_is": wf.get("mean_is_sharpe"),
        "wf_mean_oos": wf.get("mean_oos_sharpe"),
        "wf_gap": wf.get("mean_gap"),
    }


def _avg_offdiag_corr(df: pd.DataFrame) -> float:
    c = df.corr()
    n = c.shape[0]
    if n < 2:
        return float("nan")
    return float(np.nanmean(c.values[np.triu_indices(n, 1)]))


def _correlation_table(prices: pd.DataFrame, core: list[str], semis: list[str]) -> pd.DataFrame:
    rets = prices[core + semis].pct_change().dropna()
    rows = []
    for semi in semis:
        if semi not in rets.columns:
            continue
        corr = rets[core].corrwith(rets[semi]).sort_values(ascending=False)
        rows.append({
            "semi": semi,
            "mean_corr_to_core": corr.mean(),
            "max_corr_to_core": corr.max(),
            "max_corr_instrument": corr.idxmax(),
            "corr_to_SPY": corr.get("SPY", np.nan),
            "corr_to_IWM": corr.get("IWM", np.nan),
        })
    return pd.DataFrame(rows)


def main() -> int:
    core = core_symbols(expanded=False)
    print(f"Core symbols: {core}")
    print(f"Semi symbols: {SEMI_ETFS}")

    # Load all prices together so every variant shares the same calendar.
    all_syms = list(dict.fromkeys(core + SEMI_ETFS))
    print(f"\nFetching/loading prices for {len(all_syms)} symbols...")
    prices = load_prices(
        all_syms,
        start=START,
        end=END,
        source="yfinance",
        cache_tag=CACHE_TAG,
    )
    print(f"Loaded {prices.shape[1]} symbols, {len(prices)} rows, {prices.index[0].date()} to {prices.index[-1].date()}")

    # Verify semis survived cleaning.
    missing_semis = [s for s in SEMI_ETFS if s not in prices.columns]
    if missing_semis:
        print(f"ERROR: semi ETFs dropped after cleaning: {missing_semis}")
        return 1

    print("\n## Correlation of semis to core instruments")
    corr_table = _correlation_table(prices, core, SEMI_ETFS)
    print(corr_table.to_string(index=False))

    variants = [
        ("core_19", core),
        ("core + SMH", core + ["SMH"]),
        ("core + SOXX", core + ["SOXX"]),
        ("core + XSD", core + ["XSD"]),
        ("core + SMH + SOXX + XSD", core + SEMI_ETFS),
    ]

    results = []
    for name, syms in variants:
        results.append(_run_variant(name, syms, prices))

    print("\n\n## Summary comparison")
    cols = [
        "name", "n_instruments", "net_sharpe", "gross_sharpe",
        "ann_return", "ann_vol", "max_dd", "calmar",
        "idm", "mean_pairwise_corr", "diversification_ratio",
        "wf_mean_is", "wf_mean_oos", "wf_gap",
    ]
    df = pd.DataFrame(results)[cols]
    # Pretty formatting.
    formatted = df.copy()
    for c in ["net_sharpe", "gross_sharpe", "ann_vol", "max_dd", "calmar", "idm", "mean_pairwise_corr", "diversification_ratio", "wf_mean_is", "wf_mean_oos", "wf_gap"]:
        formatted[c] = formatted[c].map(_fmt)
    formatted["ann_return"] = formatted["ann_return"].map(lambda x: f"{x:.1%}" if isinstance(x, float) and not np.isnan(x) else "—")
    print(formatted.to_string(index=False))

    # Save a machine-readable report.
    report_path = os.path.join("data", "semis_evaluation.json")
    with open(report_path, "w") as f:
        json.dump(
            {
                "run_at": datetime.now(timezone.utc).isoformat(),
                "start": str(prices.index[0].date()),
                "end": str(prices.index[-1].date()),
                "correlations": corr_table.to_dict(orient="records"),
                "variants": results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nSaved report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
