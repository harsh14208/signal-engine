"""Optional Alpaca paper execution (Tier A5).

Reads the latest target, computes delta notional vs current Alpaca positions, and
submits fractional notional orders to match the target. Only runs if the kill
switch is NOT engaged. Use `--paper` (default) or explicitly `--live`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from signal_engine.live import (  # noqa: E402
    DEFAULT_KILL_SWITCH_PATH,
    DEFAULT_TARGETS_PATH,
    load_latest_target_for_book,
    read_kill_switch,
)

PAPER_BASE = "https://paper-api.alpaca.markets"
LIVE_BASE = "https://api.alpaca.markets"
ORDERS_PATH = "/v2/orders"
POSITIONS_PATH = "/v2/positions"


def _load_env(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file (no quoting logic)."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _credentials(live: bool) -> tuple[str, str]:
    env = _load_env(repo_root / ".env")
    mode = "live" if live else "paper"
    # Prefer signal-engine-DEDICATED creds (ALPACA_SE_*) so the forward-test paper
    # account stays ISOLATED from the parent project's Alpaca account — sharing one
    # account would net positions on overlapping tickers (SPY, GLD, …) and make
    # reconciliation meaningless. Falls back to the generic vars if SE_* are unset.
    key = (
        os.environ.get("ALPACA_SE_API_KEY")
        or env.get(f"alpaca_se_{mode}_api_key")
        or env.get("alpaca_se_api_key")
        or os.environ.get("ALPACA_API_KEY")
        or env.get(f"alpaca_{mode}_api_key")
        or env.get("alpaca_api_key")
    )
    secret = (
        os.environ.get("ALPACA_SE_API_SECRET")
        or env.get(f"alpaca_se_{mode}_api_secret")
        or env.get("alpaca_se_api_secret")
        or os.environ.get("ALPACA_API_SECRET")
        or env.get(f"alpaca_{mode}_api_secret")
        or env.get("alpaca_api_secret")
    )
    if not key or not secret:
        raise RuntimeError(
            f"Alpaca {mode} credentials not found — set ALPACA_SE_API_KEY / ALPACA_SE_API_SECRET "
            "(dedicated signal-engine paper account) in env or .env"
        )
    return key, secret


def _headers(key: str, secret: str) -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Content-Type": "application/json",
    }


def _request(
    base: str,
    path: str,
    key: str,
    secret: str,
    method: str = "GET",
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = f"{base}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=_headers(key, secret), method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode()
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode() if exc.fp else ""
        raise RuntimeError(f"Alpaca {method} {path} failed: {exc.code} {body_text}") from exc


def get_positions(base: str, key: str, secret: str) -> list[dict[str, Any]]:
    return _request(base, POSITIONS_PATH, key, secret, method="GET")


def get_account(base: str, key: str, secret: str) -> dict[str, Any]:
    return _request(base, "/v2/account", key, secret, method="GET")


def gross_scale_factor(
    target: dict[str, Any],
    equity: float,
    max_gross_mult: float,
    equity_buffer: float = 0.0,
    reference_equity: float | None = None,
) -> float:
    """Factor to scale target units so gross notional <= max_gross_mult * equity.

    The book is a leveraged long/short basket whose gross notional (|long| + |short|)
    naturally runs several times capital. On a paper account with finite buying power
    that blows the long leg through the limit, so we down-scale the whole book — keeping
    its long/short shape — to fit a gross-exposure budget. Returns 1.0 (no scaling) when
    the book already fits or when we can't compute a budget.

    Capital-buffer stabilisation (Fan/Zhang gross-exposure work): a raw budget of
    ``max_gross_mult * live_equity`` whipsaws — it *vanishes* when equity dips and
    *over-upsizes* when equity spikes, churning the book on noise. Two dampers:
      • ``equity_buffer`` reserves a fraction of equity (budget uses the remaining
        ``1 - equity_buffer``), leaving margin headroom.
      • ``reference_equity`` (e.g. a trailing/smoothed equity the caller persists)
        anchors the budget to the *lower* of live and reference equity, so a
        one-day spike can't upsize the book. Defaults to live equity (no damping).
    """
    if equity <= 0 or max_gross_mult <= 0:
        return 1.0
    notional = target.get("notional") or {}
    gross = sum(abs(float(v)) for v in notional.values() if v is not None)
    if gross <= 0:
        return 1.0
    anchor = equity if reference_equity is None else min(equity, float(reference_equity))
    budget = max_gross_mult * anchor * (1.0 - max(0.0, min(equity_buffer, 0.95)))
    return min(1.0, budget / gross)


def place_qty_order(
    base: str,
    key: str,
    secret: str,
    symbol: str,
    qty: int,
    side: str,
) -> dict[str, Any]:
    # WHOLE-SHARE quantity orders. Alpaca does NOT allow fractional SHORT positions,
    # and this engine takes both long and short legs — notional/fractional orders 422
    # on every short. Whole shares work for both directions.
    body = {
        "symbol": symbol.upper(),
        "qty": str(int(abs(qty))),
        "side": side.lower(),
        "type": "market",
        "time_in_force": "day",
    }
    return _request(base, ORDERS_PATH, key, secret, method="POST", body=body)


def cancel_all_orders(base: str, key: str, secret: str) -> None:
    """Cancel all open orders (so a re-run doesn't duplicate still-pending orders)."""
    try:
        _request(base, ORDERS_PATH, key, secret, method="DELETE")
    except Exception:
        pass


_shortable_cache: dict[str, bool] = {}


def is_shortable(base: str, key: str, secret: str, symbol: str) -> bool:
    """Whether Alpaca allows shorting this asset. Some currency/sector ETFs are not;
    the engine's short leg there can't be replicated on the paper account."""
    sym = symbol.upper()
    if sym not in _shortable_cache:
        try:
            asset = _request(base, f"/v2/assets/{sym}", key, secret)
            _shortable_cache[sym] = bool(asset.get("shortable", True))
        except Exception:
            _shortable_cache[sym] = True  # assume yes; the order will 422 if not
    return _shortable_cache[sym]


def _update_reference_equity(
    equity: float, path: Path, halflife_days: float
) -> float | None:
    """Maintain an EW-smoothed reference equity across runs (persisted to `path`).

    Returns the updated reference, or None when smoothing is disabled
    (halflife_days <= 0), in which case the gross cap uses live equity directly.
    """
    if halflife_days <= 0 or equity <= 0:
        return None
    prev = None
    if path.exists():
        try:
            prev = float(json.loads(path.read_text()).get("ref"))
        except Exception:
            prev = None
    alpha = 1.0 - 0.5 ** (1.0 / halflife_days)
    ref = equity if prev is None else prev + alpha * (equity - prev)
    path.write_text(json.dumps({"ref": ref, "last_equity": equity}))
    return ref


def execute_targets(
    target: dict[str, Any],
    live: bool = False,
    min_shares: int = 1,
    orders_path: Path | str = repo_root / "data" / "broker_orders.jsonl",
    kill_switch: dict[str, Any] | None = None,
    cancel_open: bool = True,
    max_gross_mult: float = 1.5,
    equity_buffer: float = 0.0,
    equity_ref_halflife: float = 0.0,
    equity_ref_path: Path | str = repo_root / "data" / "equity_ref.json",
) -> dict[str, Any]:
    orders_path = Path(orders_path)
    orders_path.parent.mkdir(parents=True, exist_ok=True)

    if kill_switch is None:
        kill_switch = read_kill_switch()
    if kill_switch.get("paused"):
        return {"submitted": [], "skipped": True, "reason": "kill_switch_engaged"}

    key, secret = _credentials(live)
    base = LIVE_BASE if live else PAPER_BASE
    # Cancel still-pending orders first so we reconcile against held positions only
    # (otherwise an after-close re-run double-submits the orders waiting at the open).
    if cancel_open:
        cancel_all_orders(base, key, secret)

    # Down-scale the (leveraged long/short) book to a gross-exposure budget so the long
    # leg doesn't blow through the paper account's buying power. Anchored to live equity.
    account = get_account(base, key, secret)
    equity = float(account.get("equity") or account.get("portfolio_value") or 0.0)
    ref_equity = _update_reference_equity(equity, Path(equity_ref_path), equity_ref_halflife)
    scale = gross_scale_factor(
        target, equity, max_gross_mult, equity_buffer=equity_buffer, reference_equity=ref_equity
    )

    positions = get_positions(base, key, secret)
    current: dict[str, float] = {}
    for pos in positions:
        sym = str(pos.get("symbol", "")).upper()
        current[sym] = float(pos.get("qty") or 0.0)  # signed; negative = short

    # Target units are share counts (ETF multiplier = 1) → trade WHOLE shares.
    target_shares = {
        k.upper(): round(float(v) * scale)
        for k, v in target.get("units", {}).items()
        if v is not None
    }
    all_syms = sorted(set(target_shares) | set(current))
    submitted: list[dict[str, Any]] = []
    ts = target.get("generated_at")

    for sym in all_syms:
        tgt = target_shares.get(sym, 0)
        cur = round(current.get(sym, 0.0))
        # Never cross zero in a single order: Alpaca rejects (403 insufficient_qty)
        # a sell that runs a long through flat into a short (or the mirror). Close to
        # flat this run; the reverse leg is opened from flat on the next run. The
        # daily loop converges the flip in one extra day.
        zero_cross_deferred = cur * tgt < 0
        if zero_cross_deferred:
            tgt = 0
        delta = tgt - cur
        if abs(delta) < min_shares:
            continue
        side = "buy" if delta > 0 else "sell"
        if side == "sell" and tgt < 0 and not is_shortable(base, key, secret, sym):
            # Non-shortable asset (e.g. some currency ETFs) — skip cleanly; the
            # no-broker shadow book remains the true reference for that short leg.
            resp: dict[str, Any] = {"skipped": "not_shortable"}
        else:
            try:
                resp = place_qty_order(base, key, secret, sym, abs(delta), side)
            except Exception as exc:
                resp = {"error": str(exc)}
        record = {
            "generated_at": ts,
            "target_date": target.get("date"),
            "symbol": sym,
            "target_shares": tgt,
            "current_shares": cur,
            "delta_shares": delta,
            "side": side,
            "live": live,
            "gross_scale": round(scale, 4),
            "zero_cross_deferred": zero_cross_deferred,
            "response": resp,
        }
        submitted.append(record)
        with open(orders_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    return {"submitted": submitted, "skipped": False, "gross_scale": scale, "equity": equity}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Execute the latest target on Alpaca (paper by default)."
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--paper", action="store_true", default=True, help="use Alpaca paper (default)"
    )
    mode.add_argument("--live", action="store_true", help="use Alpaca live account")
    p.add_argument("--targets", type=Path, default=DEFAULT_TARGETS_PATH)
    p.add_argument(
        "--min-shares",
        type=int,
        default=1,
        dest="min_shares",
        help="minimum whole-share delta to submit an order",
    )
    p.add_argument(
        "--orders-output",
        type=Path,
        default=repo_root / "data" / "broker_orders.jsonl",
    )
    p.add_argument("--kill-switch", type=Path, default=DEFAULT_KILL_SWITCH_PATH)
    p.add_argument(
        "--max-gross-mult",
        type=float,
        default=1.5,
        dest="max_gross_mult",
        help="cap gross exposure (|long|+|short| notional) at this multiple of account "
        "equity; the book is down-scaled to fit. Default 1.5x.",
    )
    p.add_argument(
        "--equity-buffer",
        type=float,
        default=0.0,
        dest="equity_buffer",
        help="reserve this fraction of equity out of the gross budget (margin headroom). "
        "Default 0.0.",
    )
    p.add_argument(
        "--equity-ref-halflife",
        type=float,
        default=0.0,
        dest="equity_ref_halflife",
        help="EW half-life (days) for a smoothed reference equity that anchors the gross "
        "cap to the lower of live/reference equity, damping spike-driven upsizing. "
        "0 disables (use live equity). Default 0.",
    )
    args = p.parse_args(argv)

    # The broker only ever trades the CHAMPION book. The last record in the targets
    # file may be a challenger (they're emitted after the champion each night) —
    # executing that would silently deploy an unpromoted config.
    target = load_latest_target_for_book(args.targets, "champion")
    if target is None:
        print("No champion target record found.", file=sys.stderr)
        return 1

    kill = read_kill_switch(args.kill_switch)
    if kill.get("paused"):
        print(f"Kill switch engaged ({kill.get('reason')}). Not submitting orders.")
        return 2

    try:
        res = execute_targets(
            target=target,
            live=args.live,
            min_shares=args.min_shares,
            orders_path=args.orders_output,
            kill_switch=kill,
            max_gross_mult=args.max_gross_mult,
            equity_buffer=args.equity_buffer,
            equity_ref_halflife=args.equity_ref_halflife,
        )
    except Exception as exc:
        print(f"execute_alpaca failed: {exc}", file=sys.stderr)
        return 1

    if res.get("skipped"):
        print("No orders submitted: kill switch is engaged.")
        return 2

    subs = res["submitted"]

    def _resp(r):
        return r.get("response") if isinstance(r.get("response"), dict) else {}

    accepted = [r for r in subs if _resp(r).get("id")]
    skipped = [r for r in subs if _resp(r).get("skipped")]
    rejected = [r for r in subs if not _resp(r).get("id") and not _resp(r).get("skipped")]
    scale = res.get("gross_scale", 1.0)
    equity = res.get("equity", 0.0)
    if scale < 1.0:
        print(
            f"Gross cap: book scaled to {scale:.1%} of target "
            f"(≤{args.max_gross_mult:g}x equity ${equity:,.0f})."
        )
    print(
        f"{len(accepted)}/{len(subs)} orders ACCEPTED "
        f"({'LIVE' if args.live else 'PAPER'}); {len(skipped)} skipped (non-shortable)."
    )
    for rec in subs:
        resp = _resp(rec)
        flag = "ok " if resp.get("id") else ("skip" if resp.get("skipped") else "ERR")
        flip = " (flip→flat; reverses next run)" if rec.get("zero_cross_deferred") else ""
        print(f"  [{flag}] {rec['symbol']:6s} {rec['side']:4s} {abs(rec['delta_shares'])} sh{flip}")
    deferred = [r for r in subs if r.get("zero_cross_deferred")]
    if deferred:
        print(f"  ↺ {len(deferred)} flip(s) closed to flat; reverse leg opens on the next run.")
    if rejected:
        print(f"  ⚠ {len(rejected)} rejected — first: {str(rejected[0]['response'])[:140]}")
    return len(rejected) > 0


if __name__ == "__main__":
    raise SystemExit(main())
