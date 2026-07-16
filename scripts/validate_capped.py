"""Validate the most robust capped candidates."""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_engine.backtest import run_backtest
from signal_engine.config import Config
from signal_engine.data import load_prices
from signal_engine.experiments import count_experiments
from signal_engine.markets import symbols as core_symbols
from signal_engine.metrics import sharpe
from signal_engine.validation import (
    block_bootstrap_sharpe,
    lo_sharpe_ci,
    placebo_sharpes,
)


def _validate(label: str, syms: list[str], prices: pd.DataFrame, cfg: Config) -> None:
    result = run_backtest(prices[syms], cfg)
    daily = result.daily_returns.dropna()
    sr = sharpe(daily)
    n_trials = max(2, count_experiments() + 1)

    print(f"\n## {label} (cap={cfg.max_gross_notional}, {len(syms)} instruments, {len(daily)} days)")
    print(f"- Net Sharpe: {sr:.2f}")
    print(f"- Ann vol: {daily.std() * 16:.1%}")
    print(f"- Max gross: {result.gross_exposure.max():.1f}x | mean gross: {result.gross_exposure.mean():.1f}x")

    lo = lo_sharpe_ci(daily, n_trials=n_trials)
    if not lo.get("insufficient"):
        print(
            f"- Lo 95% CI: [{lo['ci_low']:.2f}, {lo['ci_high']:.2f}] — "
            f"SR=0 {'✅ outside' if not lo['zero_inside'] else '⚠ INSIDE'}"
        )
        print(
            f"- Deflated Sharpe (trials={lo['n_trials']}): expected max {lo['deflated_expected_max']:.2f} "
            f"→ real {lo['sharpe']:.2f} {'✅ clears' if lo['passes_deflated'] else '⚠ FAILS'}"
        )

    mc = block_bootstrap_sharpe(daily)
    if not mc.get("insufficient"):
        print(
            f"- Block-bootstrap: P5={mc['p5']:.2f} / P50={mc['p50']:.2f} / P95={mc['p95']:.2f} "
            f"— P5 {'✅ > 0' if mc['edge_real'] else '⚠ ≤ 0'}"
        )

    pl = placebo_sharpes(
        lambda panel: run_backtest(panel, cfg).daily_returns,
        n_placebo=12,
        n_instruments=len(syms),
        n_days=min(len(daily), 2500),
    )
    print(
        f"- Placebo: 95th pct={pl['noise_floor_95']:.2f} → real {sr:.2f} "
        f"{'✅ clears' if sr > pl['noise_floor_95'] else '⚠ fails'}"
    )


def main() -> int:
    core = core_symbols(expanded=False)
    semis = ["SMH", "SOXX", "XSD"]
    prices = load_prices(
        list(set(core + semis)),
        start="2007-01-01",
        end="2026-07-10",
        source="cache",
        cache_tag="options_experiment",
    )

    _validate("baseline capped 3x", core, prices, Config(max_gross_notional=3.0))
    _validate("+ semis capped 3x", core + semis, prices, Config(max_gross_notional=3.0))
    _validate("+ semis capped 4x", core + semis, prices, Config(max_gross_notional=4.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
