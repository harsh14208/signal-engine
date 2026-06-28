"""Variance-risk-premium (VRP) sleeve from FREE CBOE vol indices.

VRP — implied volatility persistently exceeds realised — is harvested by being
SHORT volatility. The key point 0j first missed: **you do not need an options
panel.** The CBOE vol indices ARE 30-day implied vol with long, free history:

    ^VIX  → SPY   (1990+)     ^OVX  → USO   (2007+)     ^EVZ  → FXE   (2007+)
    ^RVX  → IWM   (2004+)     ^GVZ  → GLD   (2008+)

For each (vol-index → underlying ETF) pair we build a synthetic **short-vol**
price series: the daily P&L of selling one day of variance at yesterday's implied
and paying today's realised, `implied_var_{t-1} − r_t²`, normalised to a roughly
constant vol with a *lagged* EW std (no lookahead) and cumulated to a price. The
engine then trades it with the normal trend rules — so it trend-follows the
short-vol carry and cuts exposure after a vol crash — and IDM rewards its low
correlation to the directional trend book.

This is a documented *proxy*, not a variance-swap book (no term structure, no
option costs).

STATUS — PARKED (2026-06-27). The free data path is confirmed: VIX-family indices
give long-history implied vol with no options panel needed. BUT injecting a
short-vol stream as a tradable instrument **detonates the engine's vol-targeting**
— across three constructions (raw, tanh-bounded, ragged-start) the book over-levers
in calm patches and the equity curve blows up (MaxDD → −inf, spurious Sharpe 4–7).
VRP is fat-tailed; harvesting it needs a DEDICATED, position-capped short-vol sizing
path, not naive instrument injection. This module is the (correct, no-lookahead,
tested) data layer for that future work and is intentionally NOT wired into the CLI.
See todos 0j.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .macro import _load_yf_series

# vol index → underlying ETF (the ETF must be in the traded panel to form a pair)
VOL_INDEX_MAP: dict[str, str] = {
    "^VIX": "SPY",
    "^RVX": "IWM",
    "^OVX": "USO",
    "^GVZ": "GLD",
    "^EVZ": "FXE",
}

_TARGET_DAILY_VOL = 0.15 / 16.0  # normalise each sleeve to ~15% annualised vol
_NORM_SPAN = 63  # EW span for the (lagged) normalising std


def short_vol_price(implied_vix: pd.Series, underlying: pd.Series) -> pd.Series:
    """Synthetic short-volatility price for one (vol-index, underlying) pair.

    No lookahead: implied is lagged one day and the normalising std is lagged.
    """
    impl = pd.to_numeric(implied_vix, errors="coerce").reindex(underlying.index).ffill()
    impl_var = (impl / 100.0) ** 2 / 252.0  # daily variance implied by the vol index
    r = underlying.pct_change()
    realised_var = r**2
    raw = impl_var.shift(1) - realised_var  # sold at yesterday's implied, pay today's realised
    norm = raw.ewm(span=_NORM_SPAN, min_periods=20).std().shift(1)
    z = (raw / norm).replace([np.inf, -np.inf], np.nan)
    # tanh-BOUND the daily move: a short-vol stream is fat-tailed, so |ret| < target
    # keeps the realised vol ~stationary and the engine's leverage bounded.
    ret = _TARGET_DAILY_VOL * np.tanh(z)  # NaN through the warmup window
    # Start the price at the FIRST valid return and leave the warmup as NaN — a
    # flat-zero leading segment reads as a ~zero-vol instrument and makes the
    # vol-targeting over-lever and detonate. NaN = ragged start the engine handles.
    price = pd.Series(np.nan, index=ret.index)
    first = ret.first_valid_index()
    if first is not None:
        seg = ret.loc[first:].fillna(0.0)
        price.loc[first:] = 100.0 * (1.0 + seg).cumprod()
    return price


def build_vrp_sleeve(prices: pd.DataFrame, mapping: dict[str, str] | None = None) -> pd.DataFrame:
    """Synthetic short-vol prices (one column `<ETF>_VRP` per available pair).

    Vol indices are fetched free from yfinance; pairs whose underlying is absent
    or whose index fails to load are skipped.
    """
    mapping = mapping or VOL_INDEX_MAP
    start = prices.index.min().strftime("%Y-%m-%d")
    end = prices.index.max().strftime("%Y-%m-%d")
    out: dict[str, pd.Series] = {}
    for vix_sym, etf in mapping.items():
        if etf not in prices.columns:
            continue
        try:
            vix = _load_yf_series(vix_sym, start, end)
        except Exception:
            continue
        if vix is None or pd.Series(vix).dropna().empty:
            continue
        out[f"{etf}_VRP"] = short_vol_price(vix, prices[etf])
    if not out:
        return pd.DataFrame(index=prices.index)
    return pd.DataFrame(out).reindex(prices.index).ffill()
