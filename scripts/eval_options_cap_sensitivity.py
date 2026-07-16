"""Sensitivity of top candidates to gross-exposure cap level."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_engine.backtest import run_backtest
from signal_engine.config import Config
from signal_engine.data import load_prices
from signal_engine.markets import symbols as core_symbols
from signal_engine.metrics import summary


CORE = core_symbols(expanded=False)
END = "2026-07-10"
CACHE_TAG = "options_experiment"
CAPS = [None, 5.0, 4.0, 3.0, 2.5]


def _wf(prices: pd.DataFrame, cfg: Config) -> dict:
    from signal_engine.metrics import sharpe as _sharpe
    prices = prices.sort_index().dropna(how="all")
    n = len(prices)
    boundaries = np.linspace(0, n, 6, dtype=int)
    embargo = max(1, int(round(n * 0.02)))
    folds = []
    for i in range(1, 5):
        train_end = boundaries[i] - embargo
        test_start = boundaries[i]
        test_end = boundaries[i + 1]
        if train_end < 128 or test_end - test_start < 30:
            continue
        train_result = run_backtest(prices.iloc[:train_end], cfg)
        from signal_engine.backtest import run_backtest_with_params
        test_result = run_backtest_with_params(
            prices.iloc[test_start:test_end], cfg, train_result.weights, train_result.idm, train_result.fdm
        )
        folds.append(
            {
                "is_sharpe": _sharpe(train_result.daily_returns),
                "oos_sharpe": _sharpe(test_result.daily_returns),
            }
        )
    if not folds:
        return {"wf_oos": None}
    return {"wf_is": float(np.mean([f["is_sharpe"] for f in folds])), "wf_oos": float(np.mean([f["oos_sharpe"] for f in folds]))}


def _run(label: str, syms: list[str], prices: pd.DataFrame, cfg: Config) -> dict:
    result = run_backtest(prices[syms], cfg)
    s = summary(result.equity, result.daily_returns, result.turnover)
    wf = _wf(prices[syms], cfg)
    return {
        "label": label,
        "cap": cfg.max_gross_notional,
        "net_sharpe": s["sharpe"],
        "ann_vol": s["ann_vol"],
        "max_dd": s["max_drawdown"],
        "mean_gross": float(result.gross_exposure.mean()),
        "max_gross": float(result.gross_exposure.max()),
        **wf,
    }


def main() -> int:
    packs = {
        "diversifier": ["BNDX", "PFF", "AMLP", "MUB", "EMLC"],
        "semis": ["SMH", "SOXX", "XSD"],
        "qqq": ["QQQ"],
    }
    prices = load_prices(
        list(set(CORE + [s for p in packs.values() for s in p])),
        start="2007-01-01",
        end=END,
        source="cache",
        cache_tag=CACHE_TAG,
    )

    variants = [
        ("baseline", CORE, Config()),
        ("+ network momentum", CORE, Config(use_network_momentum=True)),
        ("+ semis", CORE + packs["semis"], Config()),
        ("+ QQQ", CORE + packs["qqq"], Config()),
        ("+ diversifier pack", CORE + packs["diversifier"], Config()),
        ("+ diversifier + COT + carry", CORE + packs["diversifier"], Config(use_cot=True, use_carry_proxies=True)),
        ("weight: corr-cluster", CORE, Config(weight_scheme="corr_cluster")),
        ("weight: sharpe", CORE, Config(weight_scheme="sharpe")),
    ]

    results = []
    for cap in CAPS:
        for label, syms, base_cfg in variants:
            cfg = Config(**{**base_cfg.__dict__, "max_gross_notional": cap})
            print(f"Running {label} cap={cap}...")
            results.append(_run(label, syms, prices, cfg))

    df = pd.DataFrame(results)
    print("\n\n## Cap sensitivity (WF OOS)")
    pivot = df.pivot(index="label", columns="cap", values="wf_oos")
    print(pivot.to_string())

    with open("data/options_evaluation_cap_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved data/options_evaluation_cap_sensitivity.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
