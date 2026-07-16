"""Evaluate the six new optimizations, individually and combined.

All runs use a realistic 3× gross cap + 1% financing so results are comparable to
the earlier options evaluation. Walk-forward OOS Sharpe is the honest bar.
"""

# ruff: noqa: E402

from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

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
from signal_engine.experiments import log_experiment
from signal_engine.markets import symbols as core_symbols
from signal_engine.metrics import summary


CORE = core_symbols(expanded=False)
END = "2026-07-10"
CACHE_TAG = "options_experiment"
N_WF_SPLITS = 5
CAP = 3.0
FIN_RATE = 0.01


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


def _run(label: str, syms: list[str], prices: pd.DataFrame, cfg: Config) -> dict:
    panel = prices[syms].dropna(how="all").copy()
    carry = _build_carry(panel, cfg)
    cot = _build_cot(panel, cfg)
    result = run_backtest(panel, cfg, carry=carry, cot=cot)
    log_experiment(cfg, result)
    s = summary(result.equity, result.daily_returns, result.turnover)
    wf = _purged_walk_forward_ex(panel, cfg, carry, cot)
    return {
        "label": label,
        "net_sharpe": s["sharpe"],
        "ann_vol": s["ann_vol"],
        "max_dd": s["max_drawdown"],
        "mean_gross": float(result.gross_exposure.mean()),
        "max_gross": float(result.gross_exposure.max()),
        "ann_turnover": s.get("ann_turnover"),
        "wf_is": wf.get("mean_is_sharpe"),
        "wf_oos": wf.get("mean_oos_sharpe"),
        "wf_gap": wf.get("mean_gap"),
    }


def main() -> int:
    prices = load_prices(
        CORE,
        start="2007-01-01",
        end=END,
        source="cache",
        cache_tag=CACHE_TAG,
    )

    base = Config(max_gross_notional=CAP, financing_rate=FIN_RATE)
    candidates = [
        ("baseline", base),
        ("+ calibration smooth", Config(**{**base.__dict__, "calibration_smooth": 20})),
        ("+ drawdown control", Config(**{**base.__dict__, "use_drawdown_control": True})),
        ("+ trend strength filter", Config(**{**base.__dict__, "use_trend_strength_filter": True})),
        ("+ cal smooth + drawdown", Config(**{**base.__dict__, "calibration_smooth": 20, "use_drawdown_control": True})),
        ("+ cal smooth + trend", Config(**{**base.__dict__, "calibration_smooth": 20, "use_trend_strength_filter": True})),
        ("+ drawdown + trend", Config(**{**base.__dict__, "use_drawdown_control": True, "use_trend_strength_filter": True})),
        ("+ all three", Config(**{**base.__dict__, "calibration_smooth": 20, "use_drawdown_control": True, "use_trend_strength_filter": True})),
        ("+ network momentum", Config(**{**base.__dict__, "use_network_momentum": True})),
    ]

    results = []
    for label, cfg in candidates:
        print(f"Running {label}...")
        results.append(_run(label, CORE, prices, cfg))

    df = pd.DataFrame(results)
    df = df.sort_values("wf_oos", ascending=False)
    print("\n\n## Optimization evaluation (3× cap, 1% financing)")
    print(
        df[
            ["label", "net_sharpe", "ann_vol", "max_dd", "mean_gross", "ann_turnover", "wf_is", "wf_oos", "wf_gap"]
        ].to_string(index=False)
    )

    with open("data/options_evaluation_optimizations.json", "w") as f:
        json.dump({"results": results}, f, indent=2, default=str)
    print("\nSaved data/options_evaluation_optimizations.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
