"""Price data: real (yfinance + parquet cache) and synthetic (deterministic).

The core engine depends ONLY on numpy/pandas. yfinance is imported lazily inside
the real-fetch path, so tests and the synthetic demo run with zero network and
zero extra installs.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


# ── Synthetic generator (tests + offline demo) ───────────────────────────────
def synthetic_prices(
    symbols: list[str],
    n_days: int = 4000,
    seed: int = 7,
    start: str = "2009-01-02",
) -> pd.DataFrame:
    """Deterministic multi-asset panel with *persistent trends* and *low cross-
    correlation* — the exact regime a diversified trend book is built to harvest.

    NOTE: this is a labelled test/demo data-generating process, not a claim about
    real markets. Real edge is measured by `--source yfinance`.

    Construction per instrument:
        return = beta * common_factor   (small shared loading → low corr)
               + regime_drift           (sign-persistent → trends exist)
               + idiosyncratic_noise
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)

    common = rng.normal(0, 0.005, size=n_days)  # market factor
    panel = {}
    for k, sym in enumerate(symbols):
        beta = rng.uniform(0.15, 0.55)  # modest shared loading → corr ~0.1–0.2
        idio_vol = rng.uniform(0.010, 0.020)
        # Persistent drift via a slow mean-reverting OU process on the drift itself.
        # Weak enough that per-instrument trend Sharpe is a realistic ~0.2–0.4.
        drift = np.zeros(n_days)
        d = 0.0
        theta, drift_vol = 0.02, rng.uniform(0.00006, 0.00013)
        shock = rng.normal(0, drift_vol, size=n_days)
        for t in range(n_days):
            d = (1 - theta) * d + shock[t]
            drift[t] = d
        eps = rng.normal(0, idio_vol, size=n_days)
        ret = beta * common + drift + eps
        price = 100.0 * np.cumprod(1.0 + ret)
        panel[sym] = price

    return pd.DataFrame(panel, index=dates)


def random_walk_panel(n_instruments: int, n_days: int, seed: int) -> pd.DataFrame:
    """Driftless geometric random walks — NO embedded trend or carry. Any Sharpe a
    trend strategy extracts from these is pure overfitting/noise (the placebo)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2009-01-02", periods=n_days)
    cols = {}
    for k in range(n_instruments):
        vol = rng.uniform(0.008, 0.018)
        ret = rng.normal(0.0, vol, size=n_days)  # zero drift
        cols[f"RW{k}"] = 100.0 * np.cumprod(1.0 + ret)
    return pd.DataFrame(cols, index=dates)


def synthetic_carry(symbols: list[str], index: pd.DatetimeIndex, seed: int = 11) -> pd.DataFrame:
    """Slow-moving annualised-carry series per instrument (test fixture)."""
    rng = np.random.default_rng(seed)
    out = {}
    for sym in symbols:
        level = rng.uniform(-0.04, 0.06)
        walk = np.cumsum(rng.normal(0, 0.0015, size=len(index)))
        out[sym] = level + walk
    return pd.DataFrame(out, index=index)


# ── Real data (yfinance + parquet cache) ─────────────────────────────────────
def _cache_path(tag: str) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, f"prices_{tag}.parquet")


def _fetch_yfinance(symbols: list[str], start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf  # lazy: never imported in tests/synthetic path

    raw = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False, threads=True)
    if isinstance(raw.columns, pd.MultiIndex):
        px = raw["Close"]
    else:
        px = raw[["Close"]].rename(columns={"Close": symbols[0]})
    if isinstance(px, pd.Series):
        px = px.to_frame(symbols[0])
    return px


def load_prices(
    symbols: list[str],
    start: str = "2007-01-01",
    end: str | None = None,
    source: str = "auto",
    cache_tag: str = "universe",
) -> pd.DataFrame:
    """Return an adjusted-close panel (cols=symbols, index=dates), ffilled.

    source: 'synthetic' | 'yfinance' | 'cache' | 'auto' (cache→yfinance).
    """
    if source == "synthetic":
        return synthetic_prices(symbols)

    path = _cache_path(cache_tag)
    if source in ("cache", "auto") and os.path.exists(path):
        px = pd.read_parquet(path)
        missing = [s for s in symbols if s not in px.columns]
        if not missing:
            return _clean(px.reindex(columns=symbols).dropna(how="all", axis=1))
        if source == "cache":
            raise FileNotFoundError(
                f"Cached prices at {path} missing requested symbols: {missing}"
            )

    if source in ("yfinance", "auto"):
        px = _fetch_yfinance(symbols, start, end)
        px = _clean(px)
        px.to_parquet(path)
        return px

    raise FileNotFoundError(f"No cached prices at {path} and source={source!r}")


def _clean(px: pd.DataFrame) -> pd.DataFrame:
    px = px.sort_index().ffill().dropna(how="all")
    # Drop instruments with too little history to estimate slow rules.
    keep = [c for c in px.columns if px[c].notna().sum() > 300]
    return px[keep].dropna(how="all")
