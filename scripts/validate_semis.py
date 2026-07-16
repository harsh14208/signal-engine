"""Run the statistical honesty suite on the baseline and all-semis variants."""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_engine.backtest import run_backtest
from signal_engine.config import Config
from signal_engine.data import load_prices
from signal_engine.markets import symbols as core_symbols
from signal_engine.metrics import sharpe
from signal_engine.validation import (
    block_bootstrap_sharpe,
    honest_n_trials,
    lo_sharpe_ci,
    placebo_sharpes,
)


def _validate(label: str, prices: pd.DataFrame) -> None:
    cfg = Config()
    result = run_backtest(prices, cfg)
    daily = result.daily_returns.dropna()
    sr = sharpe(daily)
    n_trials = honest_n_trials()

    print(f"\n## Validation: {label} ({prices.shape[1]} instruments, {len(daily)} days)\n")
    print(f"- Net Sharpe: {sr:.2f}")

    lo = lo_sharpe_ci(daily, n_trials=n_trials)
    if lo.get("insufficient"):
        print("- Insufficient data for Lo CI.")
    else:
        verdict = "✅ outside" if not lo["zero_inside"] else "⚠ INSIDE"
        print(
            f"- Lo (2002) 95% CI: [{lo['ci_low']:.2f}, {lo['ci_high']:.2f}] "
            f"— SR=0 is {verdict} (N={lo['n']}, trials={lo['n_trials']})"
        )
        dverdict = "✅ clears" if lo["passes_deflated"] else "⚠ FAILS"
        print(
            f"- Deflated Sharpe: expected max by chance = {lo['deflated_expected_max']:.2f} "
            f"→ real {lo['sharpe']:.2f} {dverdict} it"
        )

    mc = block_bootstrap_sharpe(daily)
    if not mc.get("insufficient"):
        edge = "✅ > 0" if mc["edge_real"] else "⚠ ≤ 0"
        print(
            f"- Block-bootstrap MC: P5={mc['p5']:.2f} / P50={mc['p50']:.2f} / P95={mc['p95']:.2f} "
            f"— P5 {edge}"
        )

    n_days = min(len(daily), 2500)
    pl = placebo_sharpes(
        lambda panel: run_backtest(panel, cfg).daily_returns,
        n_placebo=12,
        n_instruments=prices.shape[1],
        n_days=n_days,
    )
    clears = "✅ clears" if sr > pl["noise_floor_95"] else "⚠ does NOT clear"
    print(
        f"- Random-walk placebo (n={pl['n_placebo']}): noise floor "
        f"mean={pl['mean']:.2f}, 95th pct={pl['noise_floor_95']:.2f} → "
        f"real {sr:.2f} {clears} the floor"
    )


def main() -> int:
    core = core_symbols(expanded=False)
    semis = ["SMH", "SOXX", "XSD"]
    all_syms = list(dict.fromkeys(core + semis))
    prices = load_prices(
        all_syms,
        start="2007-01-01",
        end="2026-07-13",
        source="cache",
        cache_tag="semis_experiment",
    )
    _validate("core_19", prices[core])
    _validate("core + SMH + SOXX + XSD", prices[core + semis])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
