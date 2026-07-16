"""Validate the top candidates from scripts/eval_options.py with the same bar as the baseline."""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_engine.backtest import run_backtest
from signal_engine.carry_data import build_carry_panel
from signal_engine.config import Config
from signal_engine.cot_data import build_cot_forecast_panel
from signal_engine.data import load_prices
from signal_engine.experiments import count_experiments
from signal_engine.markets import symbols as core_symbols
from signal_engine.metrics import sharpe
from signal_engine.validation import (
    block_bootstrap_sharpe,
    lo_sharpe_ci,
    placebo_sharpes,
)


def _build_carry(prices: pd.DataFrame, cfg: Config) -> pd.DataFrame | None:
    if not (cfg.use_carry or cfg.use_carry_proxies):
        return None
    return build_carry_panel(prices, cfg)


def _build_cot(prices: pd.DataFrame, cfg: Config) -> pd.DataFrame | None:
    if not cfg.use_cot:
        return None
    return build_cot_forecast_panel(prices, expanded=True, tag="expanded", momentum=cfg.cot_momentum)


def _validate(label: str, syms: list[str], prices: pd.DataFrame, cfg: Config) -> None:
    panel = prices[syms].dropna(how="all")
    carry = _build_carry(panel, cfg)
    cot = _build_cot(panel, cfg)
    result = run_backtest(panel, cfg, carry=carry, cot=cot)
    daily = result.daily_returns.dropna()
    sr = sharpe(daily)
    n_trials = max(2, count_experiments() + 1)

    print(f"\n## Validation: {label} ({panel.shape[1]} instruments, {len(daily)} days)\n")
    print(f"- Net Sharpe: {sr:.2f}")

    # Leverage sanity check: max absolute notional / capital.
    max_notional = result.notional.abs().max().max()
    print(f"- Max abs notional / capital: {max_notional / cfg.capital:.1f}×")

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
        n_instruments=panel.shape[1],
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
    packs = {
        "diversifier_pack": ["BNDX", "PFF", "AMLP", "MUB", "EMLC"],
        "rate_pack": ["BNDX", "MUB"],
        "semis": ["SMH", "SOXX", "XSD"],
        "qqq": ["QQQ"],
    }
    all_syms = list(set(core + [s for p in packs.values() for s in p]))
    prices = load_prices(
        all_syms,
        start="2007-01-01",
        end="2026-07-10",
        source="cache",
        cache_tag="options_experiment",
    )

    candidates = [
        ("core_19_baseline", core, Config()),
        ("+ diversifier pack", core + packs["diversifier_pack"], Config()),
        ("+ diversifier pack + COT + carry", core + packs["diversifier_pack"], Config(use_cot=True, use_carry_proxies=True)),
        ("+ rate pack (BNDX/MUB)", core + packs["rate_pack"], Config()),
        ("weight: corr-cluster", core, Config(weight_scheme="corr_cluster")),
        ("weight: sharpe", core, Config(weight_scheme="sharpe")),
        ("+ network momentum", core, Config(use_network_momentum=True)),
        ("+ semis", core + packs["semis"], Config()),
        ("+ QQQ", core + packs["qqq"], Config()),
    ]

    for label, syms, cfg in candidates:
        try:
            _validate(label, syms, prices, cfg)
        except Exception as exc:
            print(f"\n⚠ {label} validation failed: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
