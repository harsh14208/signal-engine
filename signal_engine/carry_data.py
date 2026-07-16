"""Free carry proxies for the dormant carry rule.

Real carry needs futures term structure, but several clean legs are free right now:
  • bond carry from the yield-curve slope (FRED T10Y3M) — legacy proxy
  • real bond roll-down from the full Treasury curve (DGS2/5/10/30)
  • equity carry from trailing 12-month dividend yield (yfinance)

All fetchers cache results under `data/` so the engine stays deterministic offline
after the first run.
"""

from __future__ import annotations

import os
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd

from .config import Config
from .markets import Instrument, instrument_for

_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _cache_path(name: str) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, name)


def _fred_series(series_id: str) -> pd.Series:
    """Fetch a daily FRED series and return it as a timezone-naive Series."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
            df = pd.read_csv(resp, parse_dates=["observation_date"], index_col="observation_date")
    except Exception as exc:  # pragma: no cover - network failures handled upstream
        raise RuntimeError(f"Failed to fetch FRED {series_id}: {exc}") from exc
    s = df.iloc[:, 0].sort_index()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = series_id
    return s


def load_bond_carry(start: str | None = None, end: str | None = None) -> pd.Series:
    """Yield-curve slope (10Y - 3M) from FRED, as a decimal annual carry."""
    s = _fred_series("T10Y3M").dropna() / 100.0
    s.name = "bond_carry"
    start = pd.Timestamp(start or "1900-01-01")
    end = pd.Timestamp(end or datetime.now().strftime("%Y-%m-%d"))
    idx = pd.bdate_range(start=start, end=end)
    return s.reindex(idx, method="ffill").fillna(0.0)


def load_us_short_rate(start: str | None = None, end: str | None = None) -> pd.Series:
    """US 3-month Treasury yield (DGS3M0) from FRED, as a decimal."""
    s = _fred_series("DGS3MO").dropna() / 100.0
    s.name = "us_short_rate"
    start = pd.Timestamp(start or "1900-01-01")
    end = pd.Timestamp(end or datetime.now().strftime("%Y-%m-%d"))
    idx = pd.bdate_range(start=start, end=end)
    return s.reindex(idx, method="ffill").fillna(0.0)


def load_treasury_curve(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Daily Treasury curve (DGS2/5/10/30) from FRED, as decimals.

    Returns a DataFrame indexed by business date with columns
    {'DGS2','DGS5','DGS10','DGS30'}.
    """
    start_dt = pd.Timestamp(start or "2007-01-01")
    end_dt = pd.Timestamp(end or datetime.now().strftime("%Y-%m-%d"))
    idx = pd.bdate_range(start=start_dt, end=end_dt)

    cache = _cache_path("treasury_curve.parquet")
    if os.path.exists(cache):
        cached = pd.read_parquet(cache)
        cached.index = pd.to_datetime(cached.index).tz_localize(None)
        if cached.index.min() <= start_dt and cached.index.max() >= end_dt:
            return cached.loc[start_dt:end_dt]

    series = {}
    for sid in ["DGS2", "DGS5", "DGS10", "DGS30"]:
        s = _fred_series(sid).dropna() / 100.0
        series[sid] = s

    df = pd.DataFrame(series)
    df = df.reindex(idx, method="ffill").ffill().bfill()
    df.to_parquet(cache)
    return df


def _interpolate_yield(curve: pd.DataFrame, tenor: float) -> pd.Series:
    """Linearly interpolate yield for a non-integer tenor from DGS2/5/10/30."""
    tenors = np.array([2.0, 5.0, 10.0, 30.0])
    cols = ["DGS2", "DGS5", "DGS10", "DGS30"]
    if tenor <= tenors[0]:
        return curve[cols[0]]
    if tenor >= tenors[-1]:
        return curve[cols[-1]]
    # Find bracket
    for i in range(len(tenors) - 1):
        if tenors[i] <= tenor <= tenors[i + 1]:
            w = (tenor - tenors[i]) / (tenors[i + 1] - tenors[i])
            return (1 - w) * curve[cols[i]] + w * curve[cols[i + 1]]
    return curve[cols[-1]]  # pragma: no cover


# Tenor and (approximate) effective duration for each Treasury ETF proxy.
_BOND_CARRY_PARAMS: dict[str, dict[str, float]] = {
    "SHY": {"tenor": 2.0, "duration": 1.8},
    "IEF": {"tenor": 7.5, "duration": 7.2},
    "TIP": {"tenor": 7.5, "duration": 7.5},
    "TLT": {"tenor": 20.0, "duration": 17.0},
}


def _roll_down_carry(curve: pd.DataFrame, sym: str) -> pd.Series:
    """Approximate annual roll-down carry for a Treasury ETF proxy.

    Carry ≈ duration × (yield(tenor) - yield(tenor - 1yr)).
    """
    params = _BOND_CARRY_PARAMS[sym]
    tenor = params["tenor"]
    duration = params["duration"]
    y_t = _interpolate_yield(curve, tenor)
    y_t_minus_1 = _interpolate_yield(curve, max(tenor - 1.0, 0.5))
    return duration * (y_t - y_t_minus_1)


def _load_dividends(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch historical dividends per symbol via yfinance and cache as parquet."""
    import yfinance as yf  # optional data dependency

    cache = _cache_path("dividends.parquet")
    if os.path.exists(cache):
        cached = pd.read_parquet(cache)
        # If cache already covers requested symbols and date range, reuse it.
        cached_syms = set(cached.columns)
        if cached_syms.issuperset(symbols):
            cached = cached.sort_index()
            cached.index = pd.to_datetime(cached.index).tz_localize(None)
            if cached.index.min() <= pd.Timestamp(start) and cached.index.max() >= pd.Timestamp(
                end
            ):
                return cached.reindex(columns=symbols)

    start_dt = pd.Timestamp(start) - pd.DateOffset(years=2)
    frames: dict[str, pd.Series] = {}
    for sym in symbols:
        try:
            div = yf.Ticker(sym).dividends
            if div is None or div.empty:
                frames[sym] = pd.Series(dtype=float, name=sym)
                continue
            div = div.copy()
            div.index = pd.to_datetime(div.index).tz_localize(None)
            div = div[start_dt:end]
            frames[sym] = div
        except Exception:
            frames[sym] = pd.Series(dtype=float, name=sym)

    # Build daily panel and forward-fill.
    all_dates = pd.bdate_range(start=start_dt, end=end)
    panel = pd.DataFrame({sym: s.reindex(all_dates).fillna(0.0) for sym, s in frames.items()})
    panel.to_parquet(cache)
    return panel


def load_equity_carry(
    symbols: list[str],
    prices: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Trailing 12-month dividend yield for each symbol, aligned to `prices`."""
    if not symbols:
        return pd.DataFrame(index=prices.index)
    start = start or prices.index.min().strftime("%Y-%m-%d")
    end = end or prices.index.max().strftime("%Y-%m-%d")
    div_panel = _load_dividends(symbols, start, end)
    # Rolling 252-day sum of dividends, then divide by price.
    trailing = div_panel.rolling(window=252, min_periods=60).sum()
    carry = trailing / prices.reindex_like(trailing).replace(0.0, np.nan)
    return carry.reindex(prices.index).fillna(0.0)


# FX currency → FRED short-rate series. 3-month interbank rates are the classic
# FX carry signal; they are monthly, so we forward-fill to daily.
_FX_CURRENCY_RATES: dict[str, tuple[str, ...]] = {
    "USD": ("DGS3MO", "IR3TIB01USM156N"),
    "EUR": ("IR3TIB01EZM156N", "ECBDFR"),
    "JPY": ("IR3TIB01JPM156N",),
    "GBP": ("IR3TIB01GBM156N",),
    "AUD": ("IR3TIB01AUM156N",),
    "CAD": ("IR3TIB01CAM156N",),
}

# FX ETF → currency longed by the ETF (vs USD).
_FX_ETF_CURRENCY: dict[str, str] = {
    "FXE": "EUR",
    "FXY": "JPY",
    "FXA": "AUD",
    "FXB": "GBP",
    "FXC": "CAD",
    "UUP": "USD",
}


def _load_fx_rate(currency: str, start: str, end: str) -> pd.Series:
    """Fetch the best available short-rate series for a currency."""
    for series_id in _FX_CURRENCY_RATES[currency]:
        try:
            s = _fred_series(series_id).dropna() / 100.0
            if not s.empty:
                s.name = f"{currency}_rate"
                idx = pd.bdate_range(start=pd.Timestamp(start), end=pd.Timestamp(end))
                return s.reindex(idx, method="ffill").ffill().bfill()
        except Exception:
            continue
    raise RuntimeError(f"Could not fetch any rate series for {currency}")


def load_fx_carry(prices: pd.DataFrame) -> pd.DataFrame:
    """Rate-differential carry for FX ETFs, aligned to `prices`.

    Carry = foreign_short_rate - usd_short_rate, expressed as a decimal annual
    return.  For UUP (USD long vs a basket) we use USD - mean(available foreign).
    """
    start = prices.index.min().strftime("%Y-%m-%d")
    end = prices.index.max().strftime("%Y-%m-%d")
    idx = prices.index

    rates: dict[str, pd.Series] = {}
    for currency in set(_FX_ETF_CURRENCY.values()):
        rates[currency] = _load_fx_rate(currency, start, end).reindex(idx).ffill()

    usd_rate = rates["USD"]
    foreign_avg = pd.DataFrame([rates[c] for c in rates if c != "USD"]).T.mean(axis=1)

    out = pd.DataFrame(0.0, index=idx, columns=prices.columns)
    for sym in prices.columns:
        currency = _FX_ETF_CURRENCY.get(sym)
        if currency is None:
            continue
        if currency == "USD":
            out[sym] = usd_rate - foreign_avg
        else:
            out[sym] = rates[currency] - usd_rate
    return out


def build_carry_panel(prices: pd.DataFrame, config: Config | None = None) -> pd.DataFrame:
    """Assemble a carry DataFrame aligned with `prices`.

    Bonds get tenor-specific roll-down carry when `use_real_bond_carry` is set,
    otherwise the legacy yield-curve slope proxy; equities/real_estate/credit get
    trailing dividend yield; FX and commodities get zero.
    """
    config = config or Config()
    symbols = list(prices.columns)
    start = prices.index.min().strftime("%Y-%m-%d")
    end = prices.index.max().strftime("%Y-%m-%d")

    out = pd.DataFrame(0.0, index=prices.index, columns=symbols)

    expanded = getattr(config, "use_expanded_universe", False)
    bond_syms = [
        s
        for s in symbols
        if (instrument_for(s, expanded) or Instrument(s, "", "other")).carry_kind == "bond_slope"
    ]
    if bond_syms:
        if getattr(config, "use_real_bond_carry", False):
            curve = load_treasury_curve(start, end)
            for sym in bond_syms:
                if sym in _BOND_CARRY_PARAMS:
                    out[sym] = _roll_down_carry(curve, sym).reindex(out.index).ffill().fillna(0.0)
        else:
            bond_carry = load_bond_carry(start, end)
            bond_vals = bond_carry.reindex(out.index).ffill().fillna(0.0)
            out[bond_syms] = pd.DataFrame({s: bond_vals for s in bond_syms}, index=out.index).values

    equity_like_classes = {"equity", "real_estate", "credit"}
    equity_syms = [
        s
        for s in symbols
        if (instrument_for(s, expanded) or Instrument(s, "", "other")).asset_class
        in equity_like_classes
    ]
    if equity_syms:
        equity_carry = load_equity_carry(equity_syms, prices, start, end)
        out[equity_syms] = equity_carry.values

    if getattr(config, "use_fx_carry", False):
        fx_carry = load_fx_carry(prices)
        # FX carry replaces any dividend-yield carry for FX ETFs.
        fx_syms = [s for s in symbols if s in _FX_ETF_CURRENCY]
        if fx_syms:
            out[fx_syms] = fx_carry[fx_syms].values

    return out
