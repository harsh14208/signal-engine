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
from datetime import datetime, timezone

import pandas as pd

from .config import FORECAST_CAP
from .data import _pit_merge

_DATASET = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
COT_REVISIONS_LOG = os.path.join(_CACHE_DIR, "cot_revisions.jsonl")

COT_SCALAR = 10.0  # z-score (std≈1) → mean |forecast| ≈ 10
COT_Z_WINDOW = 756  # ~3-year rolling z (the classic "COT index" horizon)
COT_REPORT_LAG = 5  # trading-day lag — COT is released ~3 days after the as-of date

# Absolute tolerance on the (long-short)/OI ratio above which a re-fetched report
# date counts as RESTATED. The ratio is recomputed from the same underlying
# integer fields, so an unrevised re-fetch reproduces it to float precision;
# this only needs to clear that noise floor.
COT_REVISION_TOL = 1e-9

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


def _log_cot_revision_event(event: dict) -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    event = {"at": datetime.now(timezone.utc).isoformat(), **event}
    with open(COT_REVISIONS_LOG, "a") as f:
        f.write(json.dumps(event) + "\n")


def _stitch_cot_update(old: pd.DataFrame, fresh: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Point-in-time COT cache update: keep cached report dates VERBATIM, append new ones.

    CFTC restates prior weeks' data in place on occasion (late filings, corrections),
    so a naive full-overwrite refresh silently rewrites the forecast the engine
    already traded on — the same bug class the price cache had before its PIT fix
    (the 2026-07-10 TIP/IEF whipsaw; see docs/FORWARD.md). Unlike prices, the COT
    signal is already a bounded ratio (net position / open interest), not a price
    level, so there is no basis to ratio-stitch onto: new report dates are appended
    as fetched, and a fresh value that disagrees with an already-cached date is
    REJECTED and reported, not applied.

    Returns (stitched_panel, revision_report). The report lists symbols whose
    already-cached dates moved beyond `COT_REVISION_TOL` on re-fetch — the
    restatements being *rejected*.
    """
    return _pit_merge(
        old,
        fresh,
        tol=COT_REVISION_TOL,
        diff_fn=lambda o_c, f_c: (f_c - o_c).abs(),
        diff_key="max_abs_diff",
        combine_new_rows=lambda o, f, anchor, new_rows: pd.concat([o, new_rows]),
        keep_old_when_no_overlap=False,
    )


def build_cot_signal_panel(
    prices: pd.DataFrame,
    expanded: bool = False,
    use_cache: bool = True,
    tag: str | None = None,
    refresh: bool = False,
    rebase: bool = False,
) -> pd.DataFrame:
    """Daily (weekly-ffilled) net-positioning signal per mappable instrument.

    The cache is POINT-IN-TIME: a refresh never rewrites report dates already
    cached (see `_stitch_cot_update`); rejected upstream restatements are logged
    to `data/cot_revisions.jsonl`. Pass `rebase=True` to deliberately accept the
    fresh values for already-cached dates wholesale (also logged).
    """
    tag = tag or ("expanded" if expanded else "core")
    cache = os.path.join(_CACHE_DIR, f"cot_signal_{tag}.parquet")
    syms = [s for s in prices.columns if s in COT_MAP]
    if not syms:
        return pd.DataFrame(index=prices.index)

    if not refresh and use_cache and os.path.exists(cache):
        cached = pd.read_parquet(cache)
        cached.index = pd.to_datetime(cached.index)
        if set(syms).issubset(set(cached.columns)):
            return cached.reindex(prices.index).ffill()[syms]

    fresh_out: dict[str, pd.Series] = {}
    for sym in syms:
        inc, exc = COT_MAP[sym]
        try:
            s = _fetch_market(inc, exc)
        except Exception:
            continue
        if s is not None and not s.empty:
            fresh_out[sym] = s
    fresh = pd.DataFrame(fresh_out).sort_index() if fresh_out else pd.DataFrame()

    if os.path.exists(cache):
        old = pd.read_parquet(cache)
        old.index = pd.to_datetime(old.index)
        if rebase:
            panel = fresh if not fresh.empty else old
            _log_cot_revision_event({"tag": tag, "action": "rebase"})
        else:
            panel, report = _stitch_cot_update(old, fresh)
            if report["n_symbols_revised"]:
                _log_cot_revision_event({"tag": tag, "action": "stitched", **report})
                syms_r = ", ".join(sorted(report["revised"]))
                print(
                    f"⚠ upstream COT restatement REJECTED (PIT cache kept) for "
                    f"{report['n_symbols_revised']} symbol(s): {syms_r} "
                    f"→ logged to {os.path.basename(COT_REVISIONS_LOG)}"
                )
    else:
        panel = fresh

    if panel.empty:
        return pd.DataFrame(index=prices.index)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    panel.to_parquet(cache)
    out_syms = [s for s in syms if s in panel.columns]
    return panel.reindex(prices.index).ffill()[out_syms]


def cot_forecast(
    signal: pd.Series,
    window: int = COT_Z_WINDOW,
    scalar: float = COT_SCALAR,
    cap: float = FORECAST_CAP,
    lag: int = COT_REPORT_LAG,
    momentum: bool = False,
) -> pd.Series:
    """Positioning forecast. Causal + release-lagged.

    Default (PRE-SPECIFIED) is CONTRARIAN — fade crowded specs / side with commercials.
    `momentum=True` is the alternative hypothesis (follow specs); it is a research A/B,
    NOT a default to flip to on a better backtest (that would be sign-shopping).
    """
    s = signal.shift(lag)  # conservative release lag (no lookahead)
    mp = max(window // 4, 60)
    mean = s.rolling(window, min_periods=mp).mean()
    std = s.rolling(window, min_periods=mp).std()
    z = (s - mean) / std
    direction = scalar if momentum else -scalar
    return (direction * z).clip(-cap, cap)


def build_cot_forecast_panel(
    prices: pd.DataFrame,
    expanded: bool = False,
    tag: str | None = None,
    momentum: bool = False,
    refresh: bool = False,
) -> pd.DataFrame:
    """Full-history COT forecast per instrument, aligned to `prices.index`."""
    sig = build_cot_signal_panel(prices, expanded=expanded, tag=tag, refresh=refresh)
    if sig.empty:
        return pd.DataFrame(index=prices.index)
    fc = {c: cot_forecast(sig[c], momentum=momentum) for c in sig.columns}
    return pd.DataFrame(fc).reindex(prices.index)
