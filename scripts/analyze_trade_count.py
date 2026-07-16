"""Analyze trade frequency / turnover for top candidate additions.

Run after eval_options.py so prices are cached.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_engine.backtest import run_backtest
from signal_engine.config import Config
from signal_engine.data import load_prices
from signal_engine.markets import symbols as core_symbols


CORE = core_symbols(expanded=False)


def _trade_stats(result) -> dict:
    # Days with any turnover.
    traded_days = (result.turnover > 1e-12).sum()
    # Number of "trades" approximated by turnover / avg trade size is hard; use
    # days-with-trades and annual turnover as proxies.
    return {
        "days": len(result.daily_returns),
        "traded_days": int(traded_days),
        "pct_days_traded": traded_days / len(result.daily_returns),
        "ann_turnover": float(result.turnover.mean() * 256),
        "net_sharpe": float(result.daily_returns.mean() / result.daily_returns.std() * 16),
        "gross_mean": float(result.gross_exposure.mean()),
        "gross_max": float(result.gross_exposure.max()),
    }


def main() -> int:
    packs = {
        "semis": ["SMH", "SOXX", "XSD"],
        "qqq": ["QQQ"],
        "diversifier": ["BNDX", "PFF", "AMLP", "MUB", "EMLC"],
    }
    prices = load_prices(
        list(set(CORE + [s for p in packs.values() for s in p])),
        start="2007-01-01",
        end="2026-07-10",
        source="cache",
        cache_tag="options_experiment",
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

    rows = []
    for label, syms, cfg in variants:
        result = run_backtest(prices[syms], cfg)
        stats = _trade_stats(result)
        stats["label"] = label
        rows.append(stats)

    df = pd.DataFrame(rows)
    df = df.sort_values("net_sharpe", ascending=False)
    print("## Trade-frequency analysis (uncapped)\n")
    print(df.to_string(index=False))

    # Same with 3x cap.
    rows2 = []
    for label, syms, base_cfg in variants:
        cfg = Config(**{**base_cfg.__dict__, "max_gross_notional": 3.0})
        result = run_backtest(prices[syms], cfg)
        stats = _trade_stats(result)
        stats["label"] = label + " (cap=3x)"
        rows2.append(stats)

    df2 = pd.DataFrame(rows2)
    df2 = df2.sort_values("net_sharpe", ascending=False)
    print("\n\n## Trade-frequency analysis (3x gross cap)\n")
    print(df2.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
