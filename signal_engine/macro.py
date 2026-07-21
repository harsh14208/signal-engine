"""Free macro overlays (VIX, NFCI) for stress-regime de-risking."""

from __future__ import annotations

import os
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd

from .markets import asset_classes

_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _cache_path(name: str) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, name)


def load_vix(start: str | None = None, end: str | None = None) -> pd.Series:
    """CBOE VIX (^VIX) from yfinance, returned as a timezone-naive daily Series."""
    import yfinance as yf  # optional data dependency

    cache = _cache_path("vix.parquet")
    start_dt = pd.Timestamp(start or "2007-01-01")
    end_dt = pd.Timestamp(end or datetime.now().strftime("%Y-%m-%d"))

    if os.path.exists(cache):
        cached = pd.read_parquet(cache)
        cached.index = pd.to_datetime(cached.index).tz_localize(None)
        if cached.index.min() <= start_dt and cached.index.max() >= end_dt:
            return cached.loc[start_dt:end_dt].iloc[:, 0]

    raw = yf.download("^VIX", start=start_dt, end=end_dt, progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        raise RuntimeError("Failed to download ^VIX")
    # yfinance may return single- or multi-level columns.
    close = raw["Close"] if "Close" in raw.columns else raw.loc[:, ("Close", "^VIX")]
    s = close.squeeze().copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = "VIX"
    df = s.to_frame()
    df.to_parquet(cache)
    return s.loc[start_dt:end_dt]


def load_nfci(start: str | None = None, end: str | None = None) -> pd.Series:
    """Chicago Fed National Financial Conditions Index (NFCI) from FRED."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=NFCI"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
            df = pd.read_csv(resp, parse_dates=["observation_date"], index_col="observation_date")
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch NFCI: {exc}") from exc
    s = df.iloc[:, 0].sort_index()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = "NFCI"
    start_dt = pd.Timestamp(start or "2007-01-01")
    end_dt = pd.Timestamp(end or datetime.now().strftime("%Y-%m-%d"))
    idx = pd.bdate_range(start=start_dt, end=end_dt)
    return s.reindex(idx, method="ffill").fillna(0.0)


def load_vix_term_structure(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """VIX term-structure panel (^VIX, ^VIX9D, ^VIX3M) from yfinance.

    Returns a DataFrame with columns {'vix','vix3m','vix9d'} indexed by date.
    Missing series are forward-filled so the overlay degrades gracefully.
    """
    import yfinance as yf  # optional data dependency

    cache = _cache_path("vix_term.parquet")
    start_dt = pd.Timestamp(start or "2007-01-01")
    end_dt = pd.Timestamp(end or datetime.now().strftime("%Y-%m-%d"))

    if os.path.exists(cache):
        cached = pd.read_parquet(cache)
        cached.index = pd.to_datetime(cached.index).tz_localize(None)
        if cached.index.min() <= start_dt and cached.index.max() >= end_dt:
            return cached.loc[start_dt:end_dt]

    syms = ["^VIX", "^VIX9D", "^VIX3M"]
    raw = yf.download(syms, start=start_dt, end=end_dt, progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        raise RuntimeError("Failed to download VIX term-structure data")
    close = raw["Close"] if "Close" in raw.columns else raw.loc[:, ("Close", syms)]
    df = close.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.rename(columns={"^VIX": "vix", "^VIX9D": "vix9d", "^VIX3M": "vix3m"})
    df = df.ffill().dropna(how="all")
    df.to_parquet(cache)
    return df.loc[start_dt:end_dt]


def load_credit_spread(start: str | None = None, end: str | None = None) -> pd.Series:
    """Moody's Baa corporate yield minus 10-year Treasury (BAA10Y) from FRED.

    This is a long-history, free proxy for the US credit risk premium.  FRED's
    ICE BofA high-yield OAS tickers are now restricted to a rolling 3-year
    window, so BAA10Y is used instead for regime overlays that need history
    back to the 1980s.
    """
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAA10Y"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
            df = pd.read_csv(resp, parse_dates=["observation_date"], index_col="observation_date")
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch BAA10Y: {exc}") from exc
    s = df.iloc[:, 0].sort_index()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = "BAA10Y"
    start_dt = pd.Timestamp(start or "2007-01-01")
    end_dt = pd.Timestamp(end or datetime.now().strftime("%Y-%m-%d"))
    idx = pd.bdate_range(start=start_dt, end=end_dt)
    out = s.reindex(idx, method="ffill").ffill()
    return out


def _load_yf_series(symbol: str, start: str, end: str) -> pd.Series:
    """Load a single yfinance series and cache it as parquet."""
    import yfinance as yf  # optional data dependency

    cache = _cache_path(f"{symbol.lower().replace('^', '')}.parquet")
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end or datetime.now().strftime("%Y-%m-%d"))

    if os.path.exists(cache):
        cached = pd.read_parquet(cache)
        cached.index = pd.to_datetime(cached.index).tz_localize(None)
        if cached.index.min() <= start_dt and cached.index.max() >= end_dt:
            return cached.loc[start_dt:end_dt].iloc[:, 0]

    raw = yf.download(symbol, start=start_dt, end=end_dt, progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        raise RuntimeError(f"Failed to download {symbol}")
    close = raw["Close"] if "Close" in raw.columns else raw.loc[:, ("Close", symbol)]
    s = close.squeeze().copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = symbol
    s.to_frame().to_parquet(cache)
    return s.loc[start_dt:end_dt]


def _equity_drawdown(prices: pd.DataFrame) -> pd.Series:
    """Drawdown of an equal-weighted equity index from the price panel."""
    equity_symbols = set(asset_classes(expanded=True).get("equity", []))
    equity_cols = [c for c in prices.columns if c in equity_symbols]
    if not equity_cols:
        # Fall back to the first column if no column is obviously equity.
        equity_cols = [prices.columns[0]]
    eq = prices[equity_cols].mean(axis=1)
    return (eq / eq.cummax() - 1.0).fillna(0.0)


def regime_overlay(
    prices: pd.DataFrame,
    vix: pd.Series,
    nfci: pd.Series | None = None,
    vix_threshold: float = 20.0,
    drawdown_threshold: float = -0.10,
    max_degear: float = 0.5,
    vix_cap: float = 40.0,
) -> pd.Series:
    """Return a daily de-gross multiplier based on lagged VIX and equity drawdown.

    The multiplier is 1.0 in calm regimes and falls linearly toward `max_degear`
    as VIX rises above `vix_threshold` (capped at `vix_cap`) or as the equal-weighted
    equity index falls more than `drawdown_threshold` below its trailing high.
    """
    idx = prices.index
    vix_lag = vix.reindex(idx).ffill().shift(1)
    dd = _equity_drawdown(prices).shift(1)

    # VIX component: 1.0 at threshold, max_degear at vix_cap.
    vix_mult = pd.Series(1.0, index=idx)
    if vix_cap > vix_threshold:
        above = (vix_lag - vix_threshold) / (vix_cap - vix_threshold)
        vix_mult = 1.0 - above.clip(0.0, 1.0) * (1.0 - max_degear)
    vix_mult = vix_mult.clip(max_degear, 1.0)

    # Drawdown component: linearly de-gear once drawdown exceeds threshold.
    dd_mult = pd.Series(1.0, index=idx)
    if drawdown_threshold < 0:
        below = dd / drawdown_threshold  # positive fraction when dd < threshold
        dd_mult = 1.0 - below.clip(0.0, 1.0) * (1.0 - max_degear)
    dd_mult = dd_mult.clip(max_degear, 1.0)

    mult = np.minimum(vix_mult, dd_mult)
    return mult.fillna(1.0)


def vix_term_overlay(
    vix_df: pd.DataFrame,
    short_thresh: float = 1.10,
    long_thresh: float = 0.95,
    max_gear: float = 1.25,
    max_degear: float = 0.5,
) -> pd.Series:
    """Return a daily gear/de-gear multiplier from the VIX term structure.

    Uses lagged ratios:
      - vix9d / vix  > short_thresh → near-term fear spike → de-gear.
      - vix3m / vix  < long_thresh  → calm term structure → gear up.

    The idea is different from a simple VIX level overlay: the *shape* of the
    implied-vol curve captures whether stress is acute (short end spikes) or
    benign (curve in contango/backwardation).
    """
    idx = vix_df.index
    lagged = vix_df.shift(1)
    short_ratio = (lagged["vix9d"] / lagged["vix"]).reindex(idx)
    long_ratio = (lagged["vix3m"] / lagged["vix"]).reindex(idx)

    # Near-term fear: scale from 1.0 down to max_degear as ratio moves from
    # short_thresh to short_thresh + 0.20.
    stress = (short_ratio - short_thresh).clip(0.0, 0.20) / 0.20
    stress_mult = 1.0 - stress * (1.0 - max_degear)

    # Calm term structure: scale from 1.0 up to max_gear as ratio moves from
    # long_thresh down to long_thresh - 0.10.
    calm = (long_thresh - long_ratio).clip(0.0, 0.10) / 0.10
    calm_mult = 1.0 + calm * (max_gear - 1.0)

    mult = stress_mult * calm_mult
    return mult.clip(max_degear, max_gear).fillna(1.0)


def credit_overlay(
    spread: pd.Series,
    upper_thresh: float = 1.50,
    lower_thresh: float = 0.80,
    lookback: int = 1260,
    max_gear: float = 1.25,
    max_degear: float = 0.5,
) -> pd.Series:
    """Return a daily gear/de-gear multiplier from the credit risk premium.

    The spread is expressed as a ratio to its trailing `lookback`-day median.
    Ratios above `upper_thresh` indicate a stressed credit regime and scale the
    book down toward `max_degear`; ratios below `lower_thresh` indicate
    complacent/calm credit and scale the book up toward `max_gear`.
    """
    idx = spread.index
    lagged = spread.reindex(idx).ffill().shift(1)
    median = lagged.rolling(lookback, min_periods=lookback // 2).median()
    ratio = lagged / median

    # Stress de-gear: 1.0 at upper_thresh, max_degear at upper_thresh + 1.0.
    stress = (ratio - upper_thresh).clip(0.0, 1.0)
    stress_mult = 1.0 - stress * (1.0 - max_degear)

    # Calm gear-up: 1.0 at lower_thresh, max_gear at lower_thresh - 0.20.
    calm = (lower_thresh - ratio).clip(0.0, 0.20) / 0.20
    calm_mult = 1.0 + calm * (max_gear - 1.0)

    mult = stress_mult * calm_mult
    return mult.clip(max_degear, max_gear).fillna(1.0)


try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore[import-not-found]

    _HMM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _HMM_AVAILABLE = False


def hmm_regime_overlay(
    prices: pd.DataFrame,
    vix: pd.Series,
    spy: pd.Series,
    tnx: pd.Series | None = None,
    irx: pd.Series | None = None,
    train_window: int = 252,
    refit_stride: int = 21,
    bull_thresh: float = 0.75,
    bear_thresh: float = 0.70,
    trans_thresh: float = 0.15,
    bull_gear: float = 1.10,
    bear_degear: float = 0.70,
    trans_degear: float = 0.85,
    random_state: int = 42,
) -> pd.Series:
    """Daily HMM-based gear/de-gear multiplier.

    Trains a 2-state Gaussian HMM on an expanding window of
    (VIX level, SPY 20-day return, yield-curve spread, SPY 30-day realised vol).
    The state with the lower VIX mean is labelled "bull"; the other "bear".
    The posterior at the end of each window is mapped to a multiplier and held
    forward for `refit_stride` days, so there is no lookahead.

    Lookahead audit (2026-07): the classic HMM trap is scoring the book with the
    *smoothed* posterior of the whole series, which peeks at the future. Here we
    take only ``post[-1]`` — the posterior at the window edge, where the smoothed
    and filtered estimates coincide — and apply it strictly forward. Causal by
    construction; see `validation.assert_no_lookahead` for the regression guard.
    """
    if not _HMM_AVAILABLE:
        return pd.Series(1.0, index=prices.index)

    idx = prices.index
    vix = vix.reindex(idx).ffill()
    spy = spy.reindex(idx).ffill()
    spy_ret = spy.pct_change(train_window).fillna(0.0)
    rvol = spy.pct_change().rolling(30, min_periods=15).std().fillna(0.0) * np.sqrt(252)
    curve = pd.Series(0.0, index=idx)
    if tnx is not None and irx is not None:
        curve = (tnx.reindex(idx) - irx.reindex(idx)).ffill().fillna(0.0)

    mult = pd.Series(1.0, index=idx)
    dates = idx.tolist()

    for i in range(0, len(dates), refit_stride):
        start = max(0, i - train_window + 1)
        window = dates[start : i + 1]
        if len(window) < 60:
            continue

        X = np.column_stack(
            [
                vix.loc[window].values,
                spy_ret.loc[window].values,
                curve.loc[window].values,
                rvol.loc[window].values,
            ]
        )
        valid = ~np.isnan(X).any(axis=1)
        Xv = X[valid]
        if len(Xv) < 60:
            continue

        means = Xv.mean(axis=0)
        stds = Xv.std(axis=0)
        stds[stds < 1e-6] = 1.0
        Xn = (Xv - means) / stds

        model = GaussianHMM(
            n_components=2,
            covariance_type="diag",
            n_iter=25,
            tol=1e-4,
            random_state=random_state,
            init_params="mc",
            params="stmc",
            min_covar=1e-3,
        )
        model.startprob_ = np.full(2, 0.5)
        tm = np.full((2, 2), 0.05)
        np.fill_diagonal(tm, 0.95)
        model.transmat_ = tm
        try:
            with np.errstate(all="ignore"):
                model.fit(Xn)
            # Regularize degenerate transition / start probabilities.
            eps = 1e-6
            model.transmat_ = np.where(
                model.transmat_.sum(axis=1, keepdims=True) < eps,
                tm,
                model.transmat_,
            )
            model.transmat_ = model.transmat_ / model.transmat_.sum(axis=1, keepdims=True)
            if model.startprob_.sum() < eps:
                model.startprob_ = np.full(2, 0.5)
            else:
                model.startprob_ = model.startprob_ / model.startprob_.sum()

            post = model.predict_proba(Xn)
        except Exception:
            continue
        vix_means = [model.means_[s][0] for s in range(2)]
        bull_state = int(np.argmin(vix_means))
        bear_state = 1 - bull_state

        bull_prob = post[-1, bull_state]
        bear_prob = post[-1, bear_state]
        trans_risk = 1.0 - np.trace(model.transmat_) / 2.0

        if bull_prob >= bull_thresh:
            m = bull_gear
        elif bear_prob >= bear_thresh:
            m = bear_degear
        elif trans_risk > trans_thresh:
            m = trans_degear
        else:
            m = 1.0

        for fill_d in dates[i : min(len(dates), i + refit_stride)]:
            mult.loc[fill_d] = m

    return mult.ffill().fillna(1.0)
