"""Deep-dive experiments on the two most promising near-ship levers.

1. Cost-aware rebalancing of `expanded + regime` (buffer / regime smoothing).
2. Cheaper regime-overlay variants for the core baseline (VIX smoothing, thresholds).

Usage:
    .venv/bin/python scripts/dig_deeper.py
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from signal_engine.backtest import run_backtest
from signal_engine.config import Config
from signal_engine.data import load_prices
from signal_engine.macro import load_vix, regime_overlay
from signal_engine.markets import symbols
from signal_engine.metrics import (
    ann_vol,
    annual_turnover,
    max_drawdown,
    sharpe,
)


def _is_oos(daily: pd.Series, frac: float = 0.7) -> tuple[float, float]:
    n = int(len(daily) * frac)
    return sharpe(daily.iloc[:n]), sharpe(daily.iloc[n:])


def _eval(
    name: str,
    prices: pd.DataFrame,
    cfg: Config,
    regime: pd.Series | None = None,
    carry: pd.Series | None = None,
) -> dict:
    # If a custom regime is supplied we must still enable the overlay path.
    cfg = replace(cfg, use_regime_overlay=(regime is not None))
    result = run_backtest(prices, cfg, carry=carry, regime=regime)
    daily = result.daily_returns
    is_sr, oos_sr = _is_oos(daily)
    return {
        "name": name,
        "net_sr": sharpe(daily),
        "is_sr": is_sr,
        "oos_sr": oos_sr,
        "gap": is_sr - oos_sr,
        "ann_vol": ann_vol(daily),
        "max_dd": max_drawdown(result.equity),
        "turnover": annual_turnover(result.turnover),
        "idm": result.idm,
        "fdm": result.fdm,
    }


def _fmt_table(rows: list[dict]) -> str:
    df = pd.DataFrame(rows)
    df = df.sort_values("net_sr", ascending=False)
    lines = [
        "| name | Net SR | IS SR | OOS SR | gap | vol | MaxDD | turnover | IDM | FDM |",
        "|------|-------:|------:|-------:|----:|----:|------:|---------:|----:|----:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['name']} | {r['net_sr']:.2f} | {r['is_sr']:.2f} | {r['oos_sr']:.2f} | "
            f"{r['gap']:+.2f} | {r['ann_vol']:.1%} | {r['max_dd']:.1%} | {r['turnover']:.1f}x | "
            f"{r['idm']:.2f} | {r['fdm']:.2f} |"
        )
    return "\n".join(lines)


def expanded_regime_sweep() -> list[dict]:
    print("\n=== 1. expanded + regime: buffer & regime-smoothing sweep ===\n")
    syms = symbols(expanded=True)
    prices = load_prices(syms, source="cache", cache_tag="expanded")
    vix = load_vix(prices.index.min().strftime("%Y-%m-%d"), prices.index.max().strftime("%Y-%m-%d"))
    base_regime = regime_overlay(prices, vix)

    rows = []
    for buf in (0.10, 0.15, 0.20, 0.30):
        for smooth in (None, 5, 10, 20):
            rg = base_regime if smooth is None else base_regime.ewm(span=smooth, min_periods=1).mean()
            cfg = Config(
                use_expanded_universe=True,
                use_regime_overlay=False,  # pass regime manually
                buffer_fraction=buf,
            )
            name = f"buf={buf:.0%} smooth={smooth}"
            rows.append(_eval(name, prices, cfg, regime=rg))
    return rows


def regime_signal_sweep() -> list[dict]:
    print("\n=== 2. baseline: cheaper regime signal sweep ===\n")
    syms = symbols(expanded=False)
    prices = load_prices(syms, source="cache", cache_tag="universe")
    vix = load_vix(prices.index.min().strftime("%Y-%m-%d"), prices.index.max().strftime("%Y-%m-%d"))

    rows = []
    for vix_thresh in (18.0, 20.0, 22.0):
        for dd_thresh in (-0.08, -0.10, -0.12):
            for max_dg in (0.5, 0.67):
                for smooth in (None, 5, 10):
                    raw = regime_overlay(
                        prices,
                        vix,
                        vix_threshold=vix_thresh,
                        drawdown_threshold=dd_thresh,
                        max_degear=max_dg,
                    )
                    rg = raw if smooth is None else raw.ewm(span=smooth, min_periods=1).mean()
                    cfg = Config(use_regime_overlay=False, buffer_fraction=0.10)
                    name = f"vix={vix_thresh} dd={dd_thresh} degear={max_dg} smooth={smooth}"
                    rows.append(_eval(name, prices, cfg, regime=rg))
    # built-in default parameters for reference
    rows.append(_eval("built-in regime", prices, Config(), regime=regime_overlay(prices, vix)))
    rows.append(_eval("no regime", prices, Config()))
    return rows


def main() -> None:
    expanded_rows = expanded_regime_sweep()
    print(_fmt_table(expanded_rows))

    regime_rows = regime_signal_sweep()
    print(_fmt_table(regime_rows))

    print("\n=== Interpretation ===")
    best_exp = sorted(expanded_rows, key=lambda r: r["oos_sr"], reverse=True)[0]
    best_reg = sorted(regime_rows, key=lambda r: r["oos_sr"], reverse=True)[0]
    print(f"Best expanded+regime by OOS: {best_exp['name']} → OOS {best_exp['oos_sr']:.2f}, "
          f"net {best_exp['net_sr']:.2f}, turnover {best_exp['turnover']:.1f}x")
    print(f"Best regime variant by OOS: {best_reg['name']} → OOS {best_reg['oos_sr']:.2f}, "
          f"net {best_reg['net_sr']:.2f}, turnover {best_reg['turnover']:.1f}x")


if __name__ == "__main__":
    main()
