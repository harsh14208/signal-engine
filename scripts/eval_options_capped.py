"""Re-evaluate the top candidates from eval_options.py with a gross-exposure cap.

This tests whether the strongest walk-forward results survive a realistic leverage
constraint. A 3× gross-notional cap is applied post-governor; lower caps can be set
via the MAX_GROSS constant.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_engine.backtest import run_backtest
from signal_engine.carry_data import build_carry_panel
from signal_engine.config import Config
from signal_engine.cot_data import build_cot_forecast_panel
from signal_engine.data import load_prices
from signal_engine.markets import symbols as core_symbols
from signal_engine.metrics import summary


CORE = core_symbols(expanded=False)
END = "2026-07-10"
CACHE_TAG = "options_experiment"
N_WF_SPLITS = 5
MAX_GROSS = 3.0  # multiples of capital


def _build_carry(prices: pd.DataFrame, cfg: Config) -> pd.DataFrame | None:
    if not (cfg.use_carry or cfg.use_carry_proxies):
        return None
    return build_carry_panel(prices, cfg)


def _build_cot(prices: pd.DataFrame, cfg: Config) -> pd.DataFrame | None:
    if not cfg.use_cot:
        return None
    return build_cot_forecast_panel(prices, expanded=True, tag="expanded", momentum=cfg.cot_momentum)


def _purged_walk_forward_ex(
    prices: pd.DataFrame,
    cfg: Config,
    carry: pd.DataFrame | None,
    cot: pd.DataFrame | None,
    n_splits: int = N_WF_SPLITS,
    embargo_frac: float = 0.02,
) -> dict:
    from signal_engine.backtest import run_backtest, run_backtest_with_params
    from signal_engine.metrics import sharpe as _sharpe

    prices = prices.sort_index().dropna(how="all")
    n = len(prices)
    if n_splits < 2 or n < 300:
        return {"insufficient": True, "n": n}

    boundaries = np.linspace(0, n, n_splits + 1, dtype=int)
    embargo = max(1, int(round(n * embargo_frac)))
    folds = []
    min_train = 128

    for i in range(1, n_splits):
        train_end = boundaries[i] - embargo
        test_start = boundaries[i]
        test_end = boundaries[i + 1]
        if train_end < min_train or test_end - test_start < 30:
            continue

        train = prices.iloc[:train_end]
        test = prices.iloc[test_start:test_end]

        train_result = run_backtest(
            train,
            cfg,
            carry=carry.iloc[:train_end] if carry is not None else None,
            cot=cot.iloc[:train_end] if cot is not None else None,
        )
        test_result = run_backtest_with_params(
            test,
            cfg,
            train_result.weights,
            train_result.idm,
            train_result.fdm,
            carry=carry.iloc[test_start:test_end] if carry is not None else None,
            cot=cot.iloc[test_start:test_end] if cot is not None else None,
        )
        folds.append(
            {
                "train_end": str(train.index[-1]),
                "test_start": str(test.index[0]),
                "test_end": str(test.index[-1]),
                "is_sharpe": _sharpe(train_result.daily_returns),
                "oos_sharpe": _sharpe(test_result.daily_returns),
            }
        )

    if not folds:
        return {"insufficient": True, "n": n}

    gaps = [f["is_sharpe"] - f["oos_sharpe"] for f in folds]
    return {
        "n_folds": len(folds),
        "folds": folds,
        "mean_is_sharpe": float(np.mean([f["is_sharpe"] for f in folds])),
        "mean_oos_sharpe": float(np.mean([f["oos_sharpe"] for f in folds])),
        "mean_gap": float(np.mean(gaps)),
    }


def _avg_offdiag_corr(df: pd.DataFrame) -> float:
    c = df.corr()
    n = c.shape[0]
    if n < 2:
        return float("nan")
    return float(np.nanmean(c.values[np.triu_indices(n, 1)]))


def _run(label: str, syms: list[str], prices: pd.DataFrame, cfg: Config) -> dict:
    panel = prices[syms].dropna(how="all").copy()
    carry = _build_carry(panel, cfg)
    cot = _build_cot(panel, cfg)
    result = run_backtest(panel, cfg, carry=carry, cot=cot)
    s = summary(result.equity, result.daily_returns, result.turnover)
    wf = _purged_walk_forward_ex(panel, cfg, carry, cot)
    return {
        "label": label,
        "n_instruments": panel.shape[1],
        "n_days": s["n_days"],
        "net_sharpe": s["sharpe"],
        "max_dd": s["max_drawdown"],
        "ann_vol": s["ann_vol"],
        "idm": result.idm,
        "mean_corr": _avg_offdiag_corr(result.per_instrument_returns),
        "mean_gross": float(result.gross_exposure.mean()),
        "max_gross": float(result.gross_exposure.max()),
        "wf_is": wf.get("mean_is_sharpe"),
        "wf_oos": wf.get("mean_oos_sharpe"),
        "wf_gap": wf.get("mean_gap"),
    }


def main() -> int:
    packs = {
        "diversifier": ["BNDX", "PFF", "AMLP", "MUB", "EMLC"],
        "rate": ["BNDX", "MUB"],
        "semis": ["SMH", "SOXX", "XSD"],
        "qqq": ["QQQ"],
    }
    all_syms = list(set(CORE + [s for p in packs.values() for s in p]))
    prices = load_prices(
        all_syms,
        start="2007-01-01",
        end=END,
        source="cache",
        cache_tag=CACHE_TAG,
    )

    capped = Config(max_gross_notional=MAX_GROSS)
    candidates = [
        ("baseline capped", CORE, capped),
        ("+ network momentum capped", CORE, Config(use_network_momentum=True, max_gross_notional=MAX_GROSS)),
        ("+ QQQ capped", CORE + packs["qqq"], capped),
        ("+ semis capped", CORE + packs["semis"], capped),
        ("+ diversifier pack capped", CORE + packs["diversifier"], capped),
        ("+ rate pack capped", CORE + packs["rate"], capped),
        ("+ diversifier + COT + carry capped", CORE + packs["diversifier"], Config(use_cot=True, use_carry_proxies=True, max_gross_notional=MAX_GROSS)),
        ("weight: corr-cluster capped", CORE, Config(weight_scheme="corr_cluster", max_gross_notional=MAX_GROSS)),
        ("weight: sharpe capped", CORE, Config(weight_scheme="sharpe", max_gross_notional=MAX_GROSS)),
    ]

    results = []
    for label, syms, cfg in candidates:
        print(f"Running {label}...")
        results.append(_run(label, syms, prices, cfg))

    df = pd.DataFrame(results)
    df = df.sort_values("wf_oos", ascending=False)
    print(f"\n\n## Results with max_gross={MAX_GROSS:.1f}x")
    print(
        df[
            ["label", "n_instruments", "net_sharpe", "ann_vol", "max_dd", "idm", "mean_corr", "mean_gross", "max_gross", "wf_is", "wf_oos", "wf_gap"]
        ].to_string(index=False)
    )

    with open("data/options_evaluation_capped.json", "w") as f:
        json.dump({"max_gross": MAX_GROSS, "results": results}, f, indent=2, default=str)
    print("\nSaved data/options_evaluation_capped.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
