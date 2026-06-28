"""Commitments of Traders (COT) positioning — FREE from the CFTC, weekly, 1986+.

The one genuinely-orthogonal free signal a futures book hadn't tried: *who is
positioned how*, not price. We use the weekly net non-commercial (large
speculator) position, normalised by open interest, and turn it into a
**contrarian / commercial-value** forecast — fade crowded specs (i.e. side with
the informed hedgers). The sign is PRE-SPECIFIED on that economic rationale; we
do not sign-shop (that would overfit).

Integration: a per-instrument forecast that combines with EWMAC/breakout via FDM
(a rule, not a de-gross overlay — the high-value shape). The z-score is computed
on FULL history (causal) so it survives the walk-forward, which re-derives other
forecasts on each short fold.

Data: CFTC Socrata `6dca-aqww` (Legacy combined). Markets fragment across contract
code changes, so we match by name pattern and take the max-open-interest contract
per report date, then normalise by OI — robust to mini/full contract switches.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

import pandas as pd

from .config import FORECAST_CAP

_DATASET = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

COT_SCALAR = 10.0  # z-score (std≈1) → mean |forecast| ≈ 10
COT_Z_WINDOW = 756  # ~3-year rolling z (the classic "COT index" horizon)
COT_REPORT_LAG = 5  # trading-day lag — COT is released ~3 days after the as-of date

# ETF → (include name-patterns [any], exclude name-patterns [none]). Covers the
# macro core; ETFs with no clean single futures contract are omitted.
COT_MAP: dict[str, tuple[list[str], list[str]]] = {
    "SPY": (["S&P 500"], ["MICRO", "ANNUAL", "DIVIDEND", "GROWTH", "VALUE", "ESG"]),
    "IWM": (["RUSSELL 2000"], ["MICRO"]),
    "USO": (["CRUDE OIL, LIGHT"], []),
    "GLD": (["GOLD"], ["MICRO", "GOLDMAN"]),
    "SLV": (["SILVER"], ["1000 TROY", "MICRO", "GOLDMAN"]),
    "TLT": (["U.S. TREASURY BONDS"], ["ULTRA"]),
    "IEF": (["10-YEAR U.S. TREASURY"], ["ULTRA"]),
    "FXE": (["EURO FX"], ["/", "XRATE"]),
    "FXY": (["JAPANESE YEN"], ["/", "XRATE", "EURO"]),
    "UUP": (["U.S. DOLLAR INDEX"], []),
    # expanded-universe extras
    "QQQ": (["NASDAQ-100"], ["MICRO"]),
    "UNG": (["NATURAL GAS"], ["ICE", "BASIS", "PENULTIMATE", "NORTHERN", "VENTURA"]),
    "CORN": (["CORN"], ["MIDAMERICA"]),
    "WEAT": (["WHEAT-SRW"], []),
}


def _fetch_market(include: list[str], exclude: list[str]) -> pd.Series | None:
    """Net non-commercial position / open interest, per report date (max-OI contract)."""
    where = (
        "("
        + " OR ".join(f"upper(market_and_exchange_names) like '%{p.upper()}%'" for p in include)
        + ")"
    )
    q = {
        "$select": (
            "report_date_as_yyyy_mm_dd,market_and_exchange_names,open_interest_all,"
            "noncomm_positions_long_all,noncomm_positions_short_all"
        ),
        "$where": where,
        "$limit": "100000",
    }
    url = _DATASET + "?" + urllib.parse.urlencode(q)
    with urllib.request.urlopen(url, timeout=90) as r:  # noqa: S310 (trusted gov URL)
        rows = json.load(r)
    best: dict[str, tuple[float, float]] = {}
    for row in rows:
        name = str(row.get("market_and_exchange_names", "")).upper()
        if any(x.upper() in name for x in exclude):
            continue
        try:
            oi = float(row.get("open_interest_all") or 0)
            nl = float(row.get("noncomm_positions_long_all") or 0)
            ns = float(row.get("noncomm_positions_short_all") or 0)
        except (TypeError, ValueError):
            continue
        if oi <= 0:
            continue
        d = str(row.get("report_date_as_yyyy_mm_dd", ""))[:10]
        if not d:
            continue
        if d not in best or oi > best[d][0]:
            best[d] = (oi, (nl - ns) / oi)
    if not best:
        return None
    return pd.Series({pd.Timestamp(d): v[1] for d, v in best.items()}).sort_index()


def build_cot_signal_panel(
    prices: pd.DataFrame, expanded: bool = False, use_cache: bool = True
) -> pd.DataFrame:
    """Daily (weekly-ffilled) net-positioning signal per mappable instrument."""
    tag = "expanded" if expanded else "core"
    cache = os.path.join(_CACHE_DIR, f"cot_signal_{tag}.parquet")
    syms = [s for s in prices.columns if s in COT_MAP]
    if not syms:
        return pd.DataFrame(index=prices.index)

    if use_cache and os.path.exists(cache):
        cached = pd.read_parquet(cache)
        cached.index = pd.to_datetime(cached.index)
        if set(syms).issubset(set(cached.columns)):
            return cached.reindex(prices.index).ffill()[syms]

    out: dict[str, pd.Series] = {}
    for sym in syms:
        inc, exc = COT_MAP[sym]
        try:
            s = _fetch_market(inc, exc)
        except Exception:
            continue
        if s is not None and not s.empty:
            out[sym] = s
    if not out:
        return pd.DataFrame(index=prices.index)
    panel = pd.DataFrame(out).sort_index()
    os.makedirs(_CACHE_DIR, exist_ok=True)
    panel.to_parquet(cache)
    return panel.reindex(prices.index).ffill()[list(out.columns)]


def cot_forecast(
    signal: pd.Series,
    window: int = COT_Z_WINDOW,
    scalar: float = COT_SCALAR,
    cap: float = FORECAST_CAP,
    lag: int = COT_REPORT_LAG,
) -> pd.Series:
    """Contrarian positioning forecast (fade crowded specs). Causal + release-lagged."""
    s = signal.shift(lag)  # conservative release lag (no lookahead)
    mp = max(window // 4, 60)
    mean = s.rolling(window, min_periods=mp).mean()
    std = s.rolling(window, min_periods=mp).std()
    z = (s - mean) / std
    return (-scalar * z).clip(-cap, cap)  # PRE-SPECIFIED contrarian sign


def build_cot_forecast_panel(prices: pd.DataFrame, expanded: bool = False) -> pd.DataFrame:
    """Full-history COT forecast per instrument, aligned to `prices.index`."""
    sig = build_cot_signal_panel(prices, expanded=expanded)
    if sig.empty:
        return pd.DataFrame(index=prices.index)
    fc = {c: cot_forecast(sig[c]) for c in sig.columns}
    return pd.DataFrame(fc).reindex(prices.index)
