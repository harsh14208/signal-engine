"""Forward-deployment helpers: target generation, shadow book, reconciliation.

This module is the bridge between the offline backtest engine and the no-broker
shadow paper loop (Tier A). It keeps the live scripts thin and testable.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtest import BacktestResult, run_backtest
from .config import Config
from .cot_data import build_cot_forecast_panel, build_cot_signal_panel, cot_forecast
from .data import load_prices
from .feature_store import DEFAULT_SNAPSHOT_DIR, save_snapshot, snapshot_from_target
from .lineage import lineage_hash
from .markets import instrument_for, symbols
from .metrics import sharpe
from .monitor import edge_decay_report, reconcile

_DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_TARGETS_PATH = _DATA_DIR / "live_targets.jsonl"


def _slice_prices(
    prices: pd.DataFrame, start: str | None = None, end: str | None = None
) -> pd.DataFrame:
    """Honour start/end even when the loader (e.g., cache) returned a wider panel."""
    if start is not None:
        prices = prices.loc[prices.index >= pd.Timestamp(start)]
    if end is not None:
        prices = prices.loc[prices.index <= pd.Timestamp(end)]
    return prices


DEFAULT_RETURNS_PATH = _DATA_DIR / "live_returns.csv"
DEFAULT_RECON_DIR = _DATA_DIR / "reconciliation"
DEFAULT_KILL_SWITCH_PATH = _DATA_DIR / "kill_switch.json"


def _build_cot_forecast_panel_with_fallback(
    prices: pd.DataFrame,
    tag: str = "core",
    momentum: bool = False,
    refresh: bool = True,
) -> pd.DataFrame | None:
    """Refresh COT if possible, otherwise fall back to the local cache."""
    if not refresh:
        return build_cot_forecast_panel(prices, tag=tag, momentum=momentum, refresh=False)
    try:
        sig = build_cot_signal_panel(prices, tag=tag, refresh=True)
    except Exception:
        sig = build_cot_signal_panel(prices, tag=tag, refresh=False)
    if sig.empty:
        return None
    fc = {c: cot_forecast(sig[c], momentum=momentum) for c in sig.columns}
    return pd.DataFrame(fc).reindex(prices.index)


def validated_config(cot: bool = True, **overrides: Any) -> Config:
    """The validated offline config for forward deployment.

    Default: core 19 ETFs + realised-vol governor + 30% buffer.  COT is opt-in
    but recommended because it is the first free lever to clear the walk-forward.
    """
    defaults = dict(
        capital=1_000_000.0,
        vol_target=0.20,
        buffer_fraction=0.30,
        use_governor=True,
        use_breakout=True,
        use_cot=cot,
        cot_momentum=False,
        cost_bps=1.5,
        cost_scheme="flat",
        weight_scheme="equal",
    )
    defaults.update(overrides)
    return Config(**defaults)


def config_from_target(target: dict[str, Any]) -> Config:
    """Rebuild the Config used to generate a target record."""
    return validated_config(
        capital=target.get("capital", 1_000_000.0),
        vol_target=target.get("vol_target", 0.20),
        buffer_fraction=target.get("buffer_fraction", 0.30),
        use_cot=target.get("use_cot", True),
        cot_momentum=target.get("cot_momentum", False),
    )


def load_latest_target(path: Path | str = DEFAULT_TARGETS_PATH) -> dict[str, Any] | None:
    """Return the most recent target record from a JSONL file."""
    path = Path(path)
    if not path.exists():
        return None
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        return None
    return json.loads(lines[-1])


def load_all_targets(path: Path | str = DEFAULT_TARGETS_PATH) -> list[dict[str, Any]]:
    """Return all target records from a JSONL file."""
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_latest_target_for_book(
    path: Path | str = DEFAULT_TARGETS_PATH, book: str = "champion"
) -> dict[str, Any] | None:
    """Most recent target record for a given shadow book.

    Records written before the champion/challenger split have no `book` field and
    are treated as the champion, so this stays backward-compatible.
    """
    for rec in reversed(load_all_targets(path)):
        if rec.get("book", "champion") == book:
            return rec
    return None


def _clean_series_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Serialize a Series-like dict, converting NaN/inf to None or float."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            out[k] = None
        else:
            out[k] = float(v)
    return out


def latest_cot_as_of(cot: pd.DataFrame | None) -> str | None:
    """Last date with any non-missing COT forecast value."""
    if cot is None or cot.empty:
        return None
    valid = cot.dropna(how="all")
    if valid.empty:
        return None
    return valid.index[-1].strftime("%Y-%m-%d")


def build_target_record(
    result: BacktestResult,
    cfg: Config,
    cot: pd.DataFrame | None = None,
    as_of: str | None = None,
    book: str = "champion",
) -> dict[str, Any]:
    """Build a date-stamped target record from the latest backtest row.

    `book` labels which shadow book this target belongs to ("champion" for the
    live default; a second label like "challenger" for a candidate config run in
    parallel), so realised returns can later be partitioned and the challenger
    promoted only on accrued forward evidence.
    """
    target_date = result.daily_returns.index[-1].strftime("%Y-%m-%d")
    units = _clean_series_dict(result.buffered.iloc[-1].to_dict())
    notional = _clean_series_dict(result.notional.iloc[-1].to_dict())
    forecast = _clean_series_dict(result.forecasts.iloc[-1].to_dict())
    return {
        "date": target_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "book": book,
        "capital": float(cfg.capital),
        "vol_target": float(cfg.vol_target),
        "buffer_fraction": float(cfg.buffer_fraction),
        "use_cot": bool(cfg.use_cot),
        "cot_momentum": bool(cfg.cot_momentum),
        "cot_as_of": latest_cot_as_of(cot),
        "idm": float(result.idm),
        "fdm": float(result.fdm),
        "governor": float(result.governor.iloc[-1]),
        "units": units,
        "notional": notional,
        "forecast": forecast,
        "as_of": as_of or target_date,
        "lineage_hash": lineage_hash(),
    }


def generate_target(
    source: str = "auto",
    cot: bool = True,
    refresh_cot: bool = True,
    end: str | None = None,
    targets_path: Path | str = DEFAULT_TARGETS_PATH,
    book: str = "champion",
    overrides: dict[str, Any] | None = None,
    snapshot: bool = True,
    snapshot_dir: Path | str = DEFAULT_SNAPSHOT_DIR,
) -> dict[str, Any]:
    """Refresh prices, run the validated config, and append a target record.

    Idempotent per (date, book): skips if a record for the latest price date and
    the same book already exists.  Pass `book="challenger"` with `overrides`
    (e.g. ``{"use_cot": False}``) to run a candidate config as a parallel shadow
    book alongside the champion.  When `snapshot` is set, an immutable point-in-time
    feature snapshot of the decision inputs is persisted for replay (Phase 3).
    """
    targets_path = Path(targets_path)
    targets_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = validated_config(cot=cot, **(overrides or {}))
    syms = symbols(expanded=False)
    end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prices = load_prices(syms, start="2007-01-01", end=end, source=source, cache_tag="universe")
    prices = _slice_prices(prices, end=end)
    if prices.empty or len(prices) < 2:
        raise RuntimeError("No price data available for target generation")

    cot_panel = (
        _build_cot_forecast_panel_with_fallback(
            prices, tag="core", momentum=cfg.cot_momentum, refresh=refresh_cot
        )
        if cfg.use_cot
        else None
    )
    result = run_backtest(prices, cfg, cot=cot_panel)
    record = build_target_record(
        result, cfg, cot=cot_panel, as_of=prices.index[-1].strftime("%Y-%m-%d"), book=book
    )

    latest = load_latest_target_for_book(targets_path, book)
    if latest is not None and latest.get("date") == record["date"]:
        return {"skipped": True, "date": record["date"], "book": book, "path": str(targets_path)}

    with open(targets_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    if snapshot:
        save_snapshot(
            record["date"], snapshot_from_target(record, prices), book=book,
            snapshot_dir=snapshot_dir,
        )
    return {"record": record, "path": str(targets_path)}


def multipliers_for(symbols_list: list[str], expanded: bool = False) -> dict[str, float]:
    """Contract/share multipliers for the given symbols."""
    out: dict[str, float] = {}
    for sym in symbols_list:
        inst = instrument_for(sym, expanded)
        out[sym] = inst.multiplier if inst else 1.0
    return out


def compute_shadow_return(
    target: dict[str, Any],
    prices: pd.DataFrame,
) -> tuple[pd.Timestamp, float] | None:
    """Return (mark_date, live_return) for the first available close after target_date."""
    target_date = pd.Timestamp(target["date"])
    if target_date not in prices.index:
        return None
    future = prices.index[prices.index > target_date]
    if len(future) == 0:
        return None
    mark_date = future[0]
    syms = [s for s in target["units"] if s in prices.columns]
    units = pd.Series({s: (target["units"].get(s) or 0.0) for s in syms})
    mult = pd.Series({s: multipliers_for([s])[s] for s in syms})
    price_change = prices.loc[mark_date, syms] - prices.loc[target_date, syms]
    pnl = (units * price_change * mult).sum()
    capital = target.get("capital", 1_000_000.0)
    live_return = float(pnl / capital) if capital else 0.0
    return mark_date, live_return


def compute_delay_slippage(
    target: dict[str, Any],
    prices: pd.DataFrame,
    arrival_prices: pd.DataFrame,
) -> float | None:
    """Overnight 'delay cost': target-date close → the price you actually enter at.

    The shadow book decides units at the target-date close but can only transact
    at the next session's arrival price (e.g. the open). The gap between the two
    is delay cost — the executable-vs-decision slippage TCA calls out. Supplying
    an arrival-price panel (opens) makes it measurable without corrupting the
    close-to-close series that stays comparable to the backtest.
    """
    target_date = pd.Timestamp(target["date"])
    if target_date not in prices.index:
        return None
    future = prices.index[prices.index > target_date]
    if len(future) == 0 or future[0] not in arrival_prices.index:
        return None
    mark_date = future[0]
    syms = [s for s in target["units"] if s in prices.columns and s in arrival_prices.columns]
    if not syms:
        return None
    units = pd.Series({s: (target["units"].get(s) or 0.0) for s in syms})
    mult = pd.Series({s: multipliers_for([s])[s] for s in syms})
    gap = arrival_prices.loc[mark_date, syms] - prices.loc[target_date, syms]
    pnl = (units * gap * mult).sum()
    capital = target.get("capital", 1_000_000.0)
    return float(pnl / capital) if capital else 0.0


def append_shadow_return(
    target: dict[str, Any] | None,
    source: str = "auto",
    end: str | None = None,
    returns_path: Path | str = DEFAULT_RETURNS_PATH,
    prices: pd.DataFrame | None = None,
    arrival_prices: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Mark the next-day return for the latest target and append to live returns."""
    returns_path = Path(returns_path)
    returns_path.parent.mkdir(parents=True, exist_ok=True)
    if target is None:
        return {"skipped": True, "reason": "no_target"}
    book = target.get("book", "champion")

    if prices is None:
        syms = [s for s in target["units"] if s]
        end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        prices = load_prices(
            syms, start=target["date"], end=end, source=source, cache_tag="universe"
        )
        prices = _slice_prices(prices, start=target["date"], end=end)

    computed = compute_shadow_return(target, prices)
    if computed is None:
        return {"skipped": True, "reason": "no_next_day_price", "target_date": target["date"]}
    mark_date, live_return = computed
    delay_return = (
        compute_delay_slippage(target, prices, arrival_prices)
        if arrival_prices is not None
        else None
    )

    existing = pd.read_csv(returns_path, parse_dates=["date"]) if returns_path.exists() else None
    if existing is not None and len(existing):
        book_col = existing["book"] if "book" in existing.columns else "champion"
        dup = ((existing["date"] == mark_date) & (book_col == book)).any()
        if dup:
            return {
                "skipped": True,
                "reason": "already_recorded",
                "date": mark_date.strftime("%Y-%m-%d"),
                "book": book,
            }

    row = {
        "date": mark_date.strftime("%Y-%m-%d"),
        "target_date": target["date"],
        "live_return": live_return,
        "mode": "shadow",
        "use_cot": target.get("use_cot"),
        "book": book,
        "delay_return": delay_return,
    }
    # Full rewrite (read+concat+write) keeps the schema consistent as columns
    # evolve — safer than a headerless append when new fields (book, delay) appear.
    combined = pd.DataFrame([row]) if existing is None else pd.concat(
        [existing.assign(date=existing["date"].dt.strftime("%Y-%m-%d")), pd.DataFrame([row])],
        ignore_index=True,
    )
    combined.to_csv(returns_path, index=False)
    return {"record": row, "path": str(returns_path)}


def build_backtest_for_reconciliation(
    target: dict[str, Any] | None,
    source: str = "auto",
    end: str | None = None,
) -> tuple[pd.Series, BacktestResult]:
    """Return modeled daily returns + full result for the same config as the target."""
    cfg = config_from_target(target) if target is not None else validated_config()
    syms = symbols(expanded=False)
    end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prices = load_prices(syms, start="2007-01-01", end=end, source=source, cache_tag="universe")
    prices = _slice_prices(prices, end=end)
    cot = (
        build_cot_forecast_panel(prices, tag="core", momentum=cfg.cot_momentum)
        if cfg.use_cot
        else None
    )
    result = run_backtest(prices, cfg, cot=cot)
    return result.daily_returns, result


def load_live_returns(path: Path | str = DEFAULT_RETURNS_PATH) -> pd.Series:
    """Load shadow/live returns as a date-indexed Series."""
    path = Path(path)
    if not path.exists():
        return pd.Series(dtype=float, name="live_return")
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date")["live_return"].sort_index()


def load_live_return_modes(path: Path | str = DEFAULT_RETURNS_PATH) -> pd.Series:
    """Load the mode (shadow / paper / live) per live-return date."""
    path = Path(path)
    if not path.exists():
        return pd.Series(dtype=str, name="mode")
    df = pd.read_csv(path, parse_dates=["date"])
    if "mode" not in df.columns:
        return pd.Series("shadow", index=df["date"], name="mode").sort_index()
    return df.set_index("date")["mode"].sort_index()


def _book_label(df: pd.DataFrame) -> pd.Series:
    """Per-row book label, deriving one from `use_cot` for legacy rows.

    Old rows predate the champion/challenger split (FORWARD.md's manual use_cot
    partition): use_cot=True → "cot_challenger", else "champion".
    """
    if "book" in df.columns and df["book"].notna().any():
        return df["book"].fillna("champion")
    if "use_cot" in df.columns:
        return df["use_cot"].map(lambda v: "cot_challenger" if v in (True, "True", "true") else "champion")
    return pd.Series("champion", index=df.index)


def champion_challenger_report(
    path: Path | str = DEFAULT_RETURNS_PATH, min_days: int = 60, promote_margin: float = 0.20
) -> dict[str, Any]:
    """Compare shadow books by their realised forward Sharpe, and recommend a promotion.

    Splits `live_returns` by book (falling back to the use_cot partition for legacy
    rows), reports per-book n/Sharpe, and only recommends promoting a challenger
    once it has ≥ `min_days` of forward data *and* beats the champion's Sharpe by
    at least `promote_margin`. This is the disciplined "promote on forward evidence,
    not backtest" gate FORWARD.md calls for.
    """
    path = Path(path)
    if not path.exists():
        return {"insufficient": True, "reason": "no_returns"}
    df = pd.read_csv(path, parse_dates=["date"])
    if df.empty:
        return {"insufficient": True, "reason": "empty"}
    df = df.assign(book=_book_label(df).values)

    books: dict[str, dict[str, Any]] = {}
    for name, grp in df.groupby("book"):
        r = grp.set_index("date")["live_return"].sort_index()
        books[str(name)] = {
            "n": int(len(r)),
            "sharpe": float(sharpe(r)) if len(r) > 1 else None,
            "mean_daily": float(r.mean()) if len(r) else None,
        }

    champ = books.get("champion")
    recommendation = "hold"
    best_challenger = None
    if champ and champ["sharpe"] is not None:
        for name, stats in books.items():
            if name == "champion" or stats["sharpe"] is None or stats["n"] < min_days:
                continue
            if stats["sharpe"] >= champ["sharpe"] + promote_margin:
                if best_challenger is None or stats["sharpe"] > books[best_challenger]["sharpe"]:
                    best_challenger = name
        if best_challenger is not None:
            recommendation = f"promote:{best_challenger}"
    return {"books": books, "recommendation": recommendation, "min_days": min_days}


def run_reconciliation(
    target: dict[str, Any] | None = None,
    source: str = "auto",
    returns_path: Path | str = DEFAULT_RETURNS_PATH,
    recon_dir: Path | str = DEFAULT_RECON_DIR,
    kill_switch_path: Path | str = DEFAULT_KILL_SWITCH_PATH,
    alarm_floor: float = 0.0,
) -> dict[str, Any]:
    """Generate a reconciliation report, persist it, and update the kill-switch."""
    returns_path = Path(returns_path)
    recon_dir = Path(recon_dir)
    kill_switch_path = Path(kill_switch_path)
    recon_dir.mkdir(parents=True, exist_ok=True)

    modeled_series, result = build_backtest_for_reconciliation(target, source=source)
    live = load_live_returns(returns_path)
    modes = load_live_return_modes(returns_path)
    # Shadow returns are costless, so compare to modeled gross. Broker modes have
    # real costs, so compare to modeled net.
    compare_gross = (
        bool((modes.reindex(live.index).fillna("shadow") == "shadow").all())
        if not modes.empty
        else True
    )
    modeled = result.gross_returns if compare_gross else modeled_series
    rec = reconcile(live, modeled)
    edge = edge_decay_report(live, alarm_floor=alarm_floor)

    report = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "target_date": target.get("date") if target else None,
        "compare_to": "gross" if compare_gross else "net",
        "modeled_sharpe": float(sharpe(result.daily_returns)),
        "live_sharpe": float(sharpe(live)) if len(live) > 60 else None,
        "reconciliation": rec,
        "edge_decay": edge,
    }

    kill = {"paused": False}
    if not rec.get("insufficient") and not rec.get("aligned"):
        kill = {
            "paused": True,
            "reason": "tracking_error",
            "date": report["date"],
            "details": rec,
        }
    elif edge.get("alarm"):
        kill = {
            "paused": True,
            "reason": "edge_decay",
            "date": report["date"],
            "details": edge,
        }

    if kill["paused"]:
        kill_switch_path.write_text(json.dumps(kill, indent=2, default=str))

    out_path = recon_dir / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    return {"report": report, "kill_switch": kill, "path": str(out_path)}


def read_kill_switch(path: Path | str = DEFAULT_KILL_SWITCH_PATH) -> dict[str, Any]:
    """Return the current kill-switch state (paused=False if none exists)."""
    path = Path(path)
    if not path.exists():
        return {"paused": False}
    return json.loads(path.read_text())
