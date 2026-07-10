"""Research: use TradingRecommendationSystem alternative data as signal-engine overlays.

The TRS project is a mean-reversion/swing signal service on individual stocks.  We do
NOT copy its approach; instead we treat its by-product data as a market-stress/risk-on
sensor for the diversified trend book.

Overlays tested:
  1. Short-volume market fear index (2024-02 → 2026-06).
  2. TRS signal stress score from recent signals (2026-04 → 2026-06).

Usage:
    .venv/bin/python scripts/trs_overlay_research.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from signal_engine.backtest import run_backtest
from signal_engine.config import Config
from signal_engine.data import load_prices
from signal_engine.markets import symbols
from signal_engine.metrics import sharpe

TRS_ROOT = "/Users/harshv.singh/TradingRecommendationSystem"


def _build_short_volume_overlay(prices: pd.DataFrame) -> pd.Series:
    """Daily median short-volume ratio → z-score → de-gross multiplier."""
    path = os.path.join(TRS_ROOT, "analysis_output", "short_volume.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    daily = df.groupby("date")["short_volume_ratio"].median()
    idx = prices.index
    svr = daily.reindex(idx).ffill().fillna(daily.median())
    # EWMA smooth; z-score on a trailing 1-year window.
    svr_smooth = svr.ewm(span=20, min_periods=20).mean()
    rolling_mean = svr_smooth.rolling(window=252, min_periods=60).mean()
    rolling_std = svr_smooth.rolling(window=252, min_periods=60).std().replace(0.0, np.nan)
    z = ((svr_smooth - rolling_mean) / rolling_std).fillna(0.0)
    # High short-volume = fear → linearly de-gross from 1.0 down to 0.5 as z moves 0→+2.
    mult = 1.0 - z.clip(0.0, 2.0) / 2.0 * 0.5
    return mult.clip(0.5, 1.0).reindex(idx).fillna(1.0)


def _build_signal_stress_overlay(prices: pd.DataFrame) -> pd.Series:
    """Fraction of recent TRS signals that are SELL, weighted by confidence."""
    path = os.path.join(TRS_ROOT, "analysis_output", "signals_recent.csv")
    df = pd.read_csv(path)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["date"] = df["created_at"].dt.date
    df["date"] = pd.to_datetime(df["date"])
    # Drop skipped and HOLD-only noise; focus on directional recommendations.
    directional = df[
        (df["is_skipped"].astype(str) == "f") & (df["action"].isin(["BUY", "SELL"]))
    ].copy()
    if directional.empty:
        return pd.Series(1.0, index=prices.index)
    directional["conf"] = directional["confidence"].fillna(50.0)
    directional["sell_w"] = (directional["action"] == "SELL").astype(float) * directional["conf"]
    directional["buy_w"] = (directional["action"] == "BUY").astype(float) * directional["conf"]
    grp = directional.groupby("date").agg(sell_w=("sell_w", "sum"), buy_w=("buy_w", "sum"))
    grp["sell_frac"] = grp["sell_w"] / (grp["sell_w"] + grp["buy_w"])
    idx = prices.index
    sf = grp["sell_frac"].reindex(idx).ffill().fillna(0.0)
    sf_smooth = sf.ewm(span=5, min_periods=1).mean()
    # High sell fraction → stress → de-gross 1.0 → 0.5 as sell_frac goes 0.5 → 1.0.
    mult = 1.0 - (sf_smooth - 0.3).clip(0.0, 0.7) / 0.7 * 0.5
    return mult.clip(0.5, 1.0).reindex(idx).fillna(1.0)


def _eval(
    name: str,
    prices: pd.DataFrame,
    cfg: Config,
    overlay: pd.Series | None = None,
    window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> dict:
    result = run_backtest(prices, cfg, regime=overlay)
    daily = result.daily_returns
    if window is not None:
        daily = daily.loc[window[0] : window[1]]
    n = len(daily)
    split = int(n * 0.7)
    return {
        "name": name,
        "net_sr": sharpe(daily),
        "is_sr": sharpe(daily.iloc[:split]),
        "oos_sr": sharpe(daily.iloc[split:]),
        "turnover": float(result.turnover.mean() * 256),
        "max_dd": float((result.equity / result.equity.cummax() - 1.0).min()),
        "mean_mult": float(overlay.mean()) if overlay is not None else 1.0,
    }


def main() -> None:
    syms = symbols(expanded=False)
    prices = load_prices(syms, source="cache", cache_tag="universe")
    cfg = Config(buffer_fraction=0.30)
    cfg_regime = Config(buffer_fraction=0.30, use_regime_overlay=True)

    short_vol = _build_short_volume_overlay(prices)
    sig_stress = _build_signal_stress_overlay(prices)

    print("\n=== TRS-data overlay research (full sample) ===\n")
    rows = [
        _eval("baseline core (buffer 30%)", prices, cfg),
        _eval("+ short-volume overlay", prices, cfg_regime, short_vol),
        _eval("+ TRS signal-stress overlay", prices, cfg_regime, sig_stress),
    ]
    _print_rows(rows)

    # The overlays only have data for short windows; evaluate impact there.
    sv_window = (short_vol[short_vol < 1].index.min(), short_vol[short_vol < 1].index.max())
    ss_window = (sig_stress[sig_stress < 1].index.min(), sig_stress[sig_stress < 1].index.max())

    print(
        f"\n=== Short-volume overlay active window: {sv_window[0].date()} → {sv_window[1].date()} ===\n"
    )
    _print_rows(
        [
            _eval("baseline", prices, cfg, window=sv_window),
            _eval("+ short-volume overlay", prices, cfg_regime, short_vol, window=sv_window),
        ]
    )

    if ss_window[0] is not pd.NaT:
        print(
            f"\n=== TRS signal-stress overlay active window: {ss_window[0].date()} → {ss_window[1].date()} ===\n"
        )
        _print_rows(
            [
                _eval("baseline", prices, cfg, window=ss_window),
                _eval(
                    "+ TRS signal-stress overlay", prices, cfg_regime, sig_stress, window=ss_window
                ),
            ]
        )


def _print_rows(rows: list[dict]) -> None:
    print(
        f"{'name':40s} {'Net':>6s} {'IS':>6s} {'OOS':>6s} {'gap':>7s} {'turn':>7s} {'MaxDD':>7s} {'meanMult':>9s}"
    )
    for r in rows:
        print(
            f"{r['name']:40s} {r['net_sr']:6.2f} {r['is_sr']:6.2f} {r['oos_sr']:6.2f} "
            f"{r['is_sr'] - r['oos_sr']:+7.2f} {r['turnover']:7.1f}x {r['max_dd']:7.1%} {r['mean_mult']:9.2f}"
        )


if __name__ == "__main__":
    main()
