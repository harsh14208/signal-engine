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
    load_latest_target,
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
    key = os.environ.get("ALPACA_API_KEY") or env.get(f"alpaca_{mode}_api_key") or env.get("alpaca_api_key")
    secret = os.environ.get("ALPACA_API_SECRET") or env.get(f"alpaca_{mode}_api_secret") or env.get("alpaca_api_secret")
    if not key or not secret:
        raise RuntimeError(f"Alpaca {mode} credentials not found in env/.env")
    return key, secret


def _headers(key: str, secret: str) -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Content-Type": "application/json",
    }


def _request(base: str, path: str, key: str, secret: str, method: str = "GET", body: dict[str, Any] | None = None) -> dict[str, Any]:
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


def place_notional_order(
    base: str,
    key: str,
    secret: str,
    symbol: str,
    notional: float,
    side: str,
) -> dict[str, Any]:
    body = {
        "symbol": symbol.upper(),
        "notional": str(round(abs(notional), 2)),
        "side": side.lower(),
        "type": "market",
        "time_in_force": "day",
    }
    return _request(base, ORDERS_PATH, key, secret, method="POST", body=body)


def execute_targets(
    target: dict[str, Any],
    live: bool = False,
    min_notional: float = 1.0,
    orders_path: Path | str = repo_root / "data" / "broker_orders.jsonl",
    kill_switch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    orders_path = Path(orders_path)
    orders_path.parent.mkdir(parents=True, exist_ok=True)

    if kill_switch is None:
        kill_switch = read_kill_switch()
    if kill_switch.get("paused"):
        return {"submitted": [], "skipped": True, "reason": "kill_switch_engaged"}

    key, secret = _credentials(live)
    base = LIVE_BASE if live else PAPER_BASE
    positions = get_positions(base, key, secret)
    current: dict[str, float] = {}
    for pos in positions:
        sym = str(pos.get("symbol", "")).upper()
        mv = pos.get("market_value") or pos.get("current_price") or 0.0
        qty = pos.get("qty") or 0.0
        if not mv and qty:
            mv = float(qty) * float(pos.get("current_price") or 0.0)
        current[sym] = float(mv)

    target_notional = {
        k.upper(): (0.0 if v is None else float(v))
        for k, v in target.get("notional", {}).items()
        if v is not None
    }
    all_syms = sorted(set(target_notional) | set(current))
    submitted: list[dict[str, Any]] = []
    ts = target.get("generated_at")

    for sym in all_syms:
        tgt = target_notional.get(sym, 0.0)
        cur = current.get(sym, 0.0)
        delta = tgt - cur
        if abs(delta) < min_notional:
            continue
        side = "buy" if delta > 0 else "sell"
        try:
            resp = place_notional_order(base, key, secret, sym, delta, side)
        except Exception as exc:
            resp = {"error": str(exc)}
        record = {
            "generated_at": ts,
            "target_date": target.get("date"),
            "symbol": sym,
            "target_notional": tgt,
            "current_notional": cur,
            "delta": delta,
            "side": side,
            "live": live,
            "response": resp,
        }
        submitted.append(record)
        with open(orders_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    return {"submitted": submitted, "skipped": False}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Execute the latest target on Alpaca (paper by default)."
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--paper", action="store_true", default=True, help="use Alpaca paper (default)")
    mode.add_argument("--live", action="store_true", help="use Alpaca live account")
    p.add_argument("--targets", type=Path, default=DEFAULT_TARGETS_PATH)
    p.add_argument(
        "--min-notional",
        type=float,
        default=1.0,
        help="minimum dollar delta to submit an order",
    )
    p.add_argument(
        "--orders-output",
        type=Path,
        default=repo_root / "data" / "broker_orders.jsonl",
    )
    p.add_argument("--kill-switch", type=Path, default=DEFAULT_KILL_SWITCH_PATH)
    args = p.parse_args(argv)

    target = load_latest_target(args.targets)
    if target is None:
        print("No target record found.", file=sys.stderr)
        return 1

    kill = read_kill_switch(args.kill_switch)
    if kill.get("paused"):
        print(f"Kill switch engaged ({kill.get('reason')}). Not submitting orders.")
        return 2

    try:
        res = execute_targets(
            target=target,
            live=args.live,
            min_notional=args.min_notional,
            orders_path=args.orders_output,
            kill_switch=kill,
        )
    except Exception as exc:
        print(f"execute_alpaca failed: {exc}", file=sys.stderr)
        return 1

    if res.get("skipped"):
        print("No orders submitted: kill switch is engaged.")
        return 2

    print(f"Submitted {len(res['submitted'])} orders ({'LIVE' if args.live else 'PAPER'}).")
    for rec in res["submitted"]:
        print(f"  {rec['symbol']:6s} {rec['side']:4s} ${abs(rec['delta']):,.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
