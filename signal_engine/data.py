"""Price data: real (yfinance + parquet cache) and synthetic (deterministic).

The core engine depends ONLY on numpy/pandas. yfinance is imported lazily inside
the real-fetch path, so tests and the synthetic demo run with zero network and
zero extra installs.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
REVISIONS_LOG = os.path.join(_CACHE_DIR, "price_revisions.jsonl")

# Relative price change on the cache/fresh overlap above which a symbol counts as
# REVISED (yfinance back-adjusts the whole history at every ex-dividend). Loose
# enough to ignore float noise from re-downloads, tight enough to catch a single
# monthly bond-ETF dividend (~0.2-0.4%).
REVISION_TOL = 5e-4


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


def _log_revision_event(event: dict) -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    event = {"at": datetime.now(timezone.utc).isoformat(), **event}
    with open(REVISIONS_LOG, "a") as f:
        f.write(json.dumps(event) + "\n")


def _stitch_update(old: pd.DataFrame, fresh: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Point-in-time cache update: keep cached history VERBATIM, append new dates.

    yfinance `auto_adjust=True` back-adjusts the entire history at every
    ex-dividend, so a naive full-overwrite silently rewrites the past the engine
    already traded on — forecasts flip retroactively (the 2026-07-10 TIP/IEF
    whipsaw) and live-vs-backtest reconciliation compares against a history that
    never existed at decision time.

    Instead, per symbol: rows up to the last cached date are kept unchanged, and
    fresh rows after it are RATIO-STITCHED onto the cached basis
    (`old[T] * fresh[t]/fresh[T]`, T = last common valid date), so forward
    total returns are exact while the past stays immutable. The cached level
    therefore drifts above the raw quote by the dividend yield accrued since the
    last rebase — immaterial next to the 30% no-trade buffer and the broker-side
    gross rescale; run `warm_cache.py --rebase` to deliberately reset the basis.

    Returns (stitched_panel, revision_report). The report lists symbols whose
    overlap moved beyond REVISION_TOL — the revisions being *rejected*.
    """
    all_cols = list(dict.fromkeys(list(old.columns) + list(fresh.columns)))
    out: dict[str, pd.Series] = {}
    revised: dict[str, dict] = {}
    for col in all_cols:
        if col not in fresh.columns or fresh[col].dropna().empty:
            out[col] = old[col]  # fresh fetch failed/missing → keep history
            continue
        if col not in old.columns or old[col].dropna().empty:
            out[col] = fresh[col]  # brand-new instrument → take as-is
            continue
        o, f = old[col].dropna(), fresh[col].dropna()
        common = o.index.intersection(f.index)
        if len(common) == 0:
            out[col] = old[col]
            continue
        anchor = common.max()
        rel = (f.loc[common] / o.loc[common] - 1.0).abs()
        if float(rel.max()) > REVISION_TOL:
            revised[col] = {
                "max_rel_diff": float(rel.max()),
                "n_dates_revised": int((rel > REVISION_TOL).sum()),
                "first_revised": str(rel[rel > REVISION_TOL].index.min().date()),
            }
        new_rows = f.loc[f.index > anchor]
        stitched = new_rows * (o.loc[anchor] / f.loc[anchor])
        out[col] = pd.concat([o, stitched])
    panel = pd.DataFrame(out).sort_index()
    report = {"revised": revised, "n_symbols_revised": len(revised)}
    return panel, report


def load_prices(
    symbols: list[str],
    start: str = "2007-01-01",
    end: str | None = None,
    source: str = "auto",
    cache_tag: str = "universe",
    rebase: bool = False,
) -> pd.DataFrame:
    """Return an adjusted-close panel (cols=symbols, index=dates), ffilled.

    source: 'synthetic' | 'yfinance' | 'cache' | 'auto' (cache→yfinance).

    The cache is POINT-IN-TIME: a yfinance refresh never rewrites dates already
    cached — new rows are ratio-stitched on (see `_stitch_update`), and rejected
    upstream revisions are logged to `data/price_revisions.jsonl`. Pass
    `rebase=True` to deliberately discard the cached basis and accept the fresh
    adjusted history wholesale (also logged).
    """
    if source == "synthetic":
        return synthetic_prices(symbols)

    path = _cache_path(cache_tag)
    if source in ("cache", "auto") and os.path.exists(path):
        px = pd.read_parquet(path)
        missing = [s for s in symbols if s not in px.columns]
        if not missing:
            px = px.reindex(columns=symbols).dropna(how="all", axis=1).sort_index()
            # Honor `end` against the cache (enables historical backfill/replay — as-of
            # targets for past dates without re-fetching). `start` is intentionally NOT
            # applied: callers want full history up to `end`, and truncating the start
            # would starve the slow rules / trip the >300-bar filter in _clean.
            if end:
                px = px.loc[:end]
            return _clean(px)
        if source == "cache":
            raise FileNotFoundError(f"Cached prices at {path} missing requested symbols: {missing}")

    if source in ("yfinance", "auto"):
        fresh = _fetch_yfinance(symbols, start, end)
        if os.path.exists(path) and not rebase:
            old = pd.read_parquet(path)
            px, report = _stitch_update(old, fresh)
            if report["n_symbols_revised"]:
                _log_revision_event(
                    {"cache_tag": cache_tag, "action": "stitched", **report}
                )
                syms_r = ", ".join(sorted(report["revised"]))
                print(
                    f"⚠ upstream price revision REJECTED (PIT cache kept) for "
                    f"{report['n_symbols_revised']} symbol(s): {syms_r} "
                    f"→ logged to {os.path.basename(REVISIONS_LOG)}"
                )
        else:
            px = fresh
            if os.path.exists(path) and rebase:
                _log_revision_event({"cache_tag": cache_tag, "action": "rebase"})
        px = _clean(px)
        px.to_parquet(path)
        return px

    raise FileNotFoundError(f"No cached prices at {path} and source={source!r}")


def _clean(px: pd.DataFrame) -> pd.DataFrame:
    px = px.sort_index().ffill().dropna(how="all")
    # Drop instruments with too little history to estimate slow rules.
    keep = [c for c in px.columns if px[c].notna().sum() > 300]
    return px[keep].dropna(how="all")
