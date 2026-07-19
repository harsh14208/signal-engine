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

from signal_engine.evaluator import (  # noqa: E402
    TradeEvaluator,
    build_evaluation_context,
    format_evaluation_for_logging,
    make_evaluator,
)
from signal_engine.live import (  # noqa: E402
    DEFAULT_KILL_SWITCH_PATH,
    DEFAULT_TARGETS_PATH,
    load_latest_target_for_book,
    read_kill_switch,
)

DEFAULT_AI_EVALUATIONS_PATH = repo_root / "data" / "ai_evaluations.jsonl"

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
    use_cash_balance: bool = False,
    ai_evaluator: TradeEvaluator | None = None,
    ai_mode: str = "scale",
    ai_evaluations_path: Path | str = DEFAULT_AI_EVALUATIONS_PATH,
) -> dict[str, Any]:
    orders_path = Path(orders_path)
    orders_path.parent.mkdir(parents=True, exist_ok=True)

    if kill_switch is None:
        kill_switch = read_kill_switch()
    if kill_switch.get("paused"):
        return {"submitted": [], "skipped": True, "reason": "kill_switch_engaged"}

    # Idempotency guard: if this exact target (same target_date + generated_at)
    # has already been executed and logged, don't submit again. Without this,
    # any accidental re-invocation for the same target — a scheduler misfire,
    # a manual re-run, a supervisor restart after a slow-but-successful prior
    # run — re-submits every order from scratch. This isn't hypothetical: the
    # broker_orders.jsonl log shows one real batch (2026-07-10) with 57
    # records instead of 19, because 51 non-skipped orders were each
    # independently submitted to Alpaca three times (three distinct order
    # IDs per symbol, not just duplicate log lines) — silently tripling
    # position sizes in the paper book that day. `live=False` for that batch
    # meant no real capital was at risk, but the same gap applies to live
    # runs, and even in paper mode it corrupts the book being used to
    # validate the strategy.
    target_date = target.get("date")
    target_generated_at = target.get("generated_at")
    if target_date and target_generated_at and orders_path.exists():
        with open(orders_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    prior = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if prior.get("target_date") == target_date and prior.get("generated_at") == target_generated_at:
                    return {
                        "submitted": [],
                        "skipped": True,
                        "reason": "already_executed",
                        "target_date": target_date,
                        "generated_at": target_generated_at,
                    }

    key, secret = _credentials(live)
    base = LIVE_BASE if live else PAPER_BASE
    # Cancel still-pending orders first so we reconcile against held positions only
    # (otherwise an after-close re-run double-submits the orders waiting at the open).
    if cancel_open:
        cancel_all_orders(base, key, secret)

    # Down-scale the (leveraged long/short) book to a gross-exposure budget so the long
    # leg doesn't blow through the paper account's buying power. By default anchored to
    # live equity; with --use-cash-balance, anchor to cash instead so the LONG leg is
    # fully cash-paid and total gross exposure is capped more conservatively. Note this
    # does NOT eliminate margin usage outright: any short leg (routine for this
    # long/short trend book) still draws Reg-T margin buying power to hold the short,
    # regardless of what the notional cap is anchored to.
    account = get_account(base, key, secret)
    equity = float(account.get("equity") or account.get("portfolio_value") or 0.0)
    cash = float(account.get("cash") or equity)
    anchor = cash if use_cash_balance else equity
    ref_equity = _update_reference_equity(anchor, Path(equity_ref_path), equity_ref_halflife)
    scale = gross_scale_factor(
        target, anchor, max_gross_mult, equity_buffer=equity_buffer, reference_equity=ref_equity
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

    # AI trade evaluation: review the proposed portfolio before any orders go out.
    # Advisory only — this can resize the target book (ai_mode="scale") or just
    # log its assessment (ai_mode="advisory"), but it can never skip or block a
    # rebalance outright. There used to be an ai_mode="block" path that returned
    # early with zero orders submitted on an AI "reject" decision; it's removed
    # (see the ai_blocked handling this replaced, in prior versions of this
    # function) so AI guidance can never prevent trades from executing.
    evaluation = None
    ai_scale = 1.0
    if ai_evaluator is not None:
        order_deltas = {
            sym: (target_shares.get(sym, 0) - round(current.get(sym, 0.0)))
            for sym in all_syms
        }
        engine_state = {
            "idm": target.get("idm"),
            "fdm": target.get("fdm"),
            "governor": target.get("governor"),
            "buffer_fraction": target.get("buffer_fraction"),
            "vol_target": target.get("vol_target"),
            "use_cot": target.get("use_cot"),
        }
        risk = {
            "gross_notional": sum(abs(float(v)) for v in (target.get("notional") or {}).values() if v is not None),
            "gross_scale_applied": round(scale, 4),
            "n_symbols": len(target_shares),
        }
        context = build_evaluation_context(
            target=target,
            current_positions=current,
            order_deltas=order_deltas,
            engine_state=engine_state,
            risk=risk,
            mode=ai_mode,
        )
        evaluation = ai_evaluator.evaluate(context)
        ai_scale = evaluation.scale if ai_mode == "scale" else 1.0
        # evaluation.decision ("approve"/"reject") is intentionally not read
        # here — it's advisory, logged below via _log_ai_evaluation, and has
        # no effect on whether orders go out. Only ai_scale (when ai_mode ==
        # "scale") ever changes what gets submitted.

        if ai_mode == "scale" and ai_scale != 1.0:
            target_shares = {k: round(v * ai_scale) for k, v in target_shares.items()}
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
            "ai_scale": round(ai_scale, 4) if evaluation is not None else None,
            "ai_mode": ai_mode if evaluation is not None else None,
            "ai_evaluation": format_evaluation_for_logging(evaluation) if evaluation is not None else None,
            "zero_cross_deferred": zero_cross_deferred,
            "response": resp,
        }
        submitted.append(record)
        with open(orders_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    if evaluation is not None:
        _log_ai_evaluation(
            ai_evaluations_path, target, evaluation, ai_mode, ai_scale=round(ai_scale, 4), applied=True
        )

    return {
        "submitted": submitted,
        "skipped": False,
        "gross_scale": scale,
        "ai_scale": round(ai_scale, 4) if evaluation is not None else 1.0,
        "ai_evaluation": format_evaluation_for_logging(evaluation) if evaluation is not None else None,
        "equity": equity,
        "cash": cash,
        "anchor": anchor,
        "anchor_basis": "cash" if use_cash_balance else "equity",
    }


def _log_ai_evaluation(
    path: Path | str,
    target: dict[str, Any],
    evaluation: Any,
    mode: str,
    ai_scale: float,
    applied: bool,
) -> None:
    """Append a structured AI evaluation record to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "generated_at": target.get("generated_at"),
        "target_date": target.get("date"),
        "book": target.get("book", "champion"),
        "mode": mode,
        "ai_scale": ai_scale,
        "applied": applied,
        "evaluation": format_evaluation_for_logging(evaluation),
    }
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


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
    p.add_argument(
        "--use-cash-balance",
        action="store_true",
        default=False,
        dest="use_cash_balance",
        help="Size the book's notional cap against cash balance instead of equity, so "
        "the long leg is fully cash-paid and the book is smaller/more conservative. "
        "Does not eliminate Reg-T margin usage on short legs, which are routine for "
        "this long/short trend book.",
    )
    p.add_argument(
        "--no-ai-evaluate",
        action="store_true",
        default=False,
        dest="no_ai_evaluate",
        help="Disable AI trade evaluation (enabled by default if an API key is available).",
    )
    p.add_argument(
        "--ai-mode",
        default="scale",
        choices=["advisory", "scale"],
        dest="ai_mode",
        help="How the AI evaluation affects execution: advisory=log only, scale=multiply "
        "target sizes. AI guidance never blocks or skips a rebalance outright — there "
        "is no mode that does that. Default: scale.",
    )
    p.add_argument(
        "--ai-provider",
        default="kimi",
        choices=["kimi", "openai", "openai-compatible"],
        dest="ai_provider",
        help="AI provider. 'kimi' uses https://api.moonshot.cn/v1; 'openai' uses OpenAI; "
        "'openai-compatible' is a generic base_url endpoint. Default: kimi.",
    )
    p.add_argument(
        "--ai-model",
        default="moonshot-v1-8k",
        dest="ai_model",
        help="Model name for the AI provider. Default: moonshot-v1-8k.",
    )
    p.add_argument(
        "--ai-api-key",
        default=None,
        dest="ai_api_key",
        help="API key for the AI provider. If omitted, reads KIMI_API_KEY / OPENAI_API_KEY "
        "from the environment or .env file.",
    )
    p.add_argument(
        "--ai-api-base",
        default=None,
        dest="ai_api_base",
        help="Override the provider base URL.",
    )
    p.add_argument(
        "--ai-required",
        action="store_true",
        default=False,
        dest="ai_required",
        help="If the AI evaluation fails and this flag is set, abort execution. "
        "Otherwise a failed call falls back to no-op (approve, scale=1.0).",
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

    ai_evaluator = None
    if not args.no_ai_evaluate:
        ai_evaluator = make_evaluator(
            use=True,
            provider=args.ai_provider,
            api_key=args.ai_api_key,
            api_base=args.ai_api_base,
            model=args.ai_model,
            required=args.ai_required,
        )

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
            use_cash_balance=args.use_cash_balance,
            ai_evaluator=ai_evaluator,
            ai_mode=args.ai_mode,
        )
    except Exception as exc:
        print(f"execute_alpaca failed: {exc}", file=sys.stderr)
        return 1

    if res.get("skipped"):
        if res.get("reason") == "ai_rejected":
            ai_eval = res.get("ai_evaluation") or {}
            print(
                f"No orders submitted: AI rejected the rebalance "
                f"(confidence={ai_eval.get('confidence')}, "
                f"reasoning={ai_eval.get('reasoning')!r})."
            )
            return 2
        print("No orders submitted: kill switch is engaged.")
        return 2

    subs = res["submitted"]

    def _resp(r):
        return r.get("response") if isinstance(r.get("response"), dict) else {}

    accepted = [r for r in subs if _resp(r).get("id")]
    skipped = [r for r in subs if _resp(r).get("skipped")]
    rejected = [r for r in subs if not _resp(r).get("id") and not _resp(r).get("skipped")]
    scale = res.get("gross_scale", 1.0)
    ai_scale = res.get("ai_scale", 1.0)
    anchor = res.get("anchor", res.get("equity", 0.0))
    anchor_basis = res.get("anchor_basis", "equity")
    if scale < 1.0:
        print(
            f"Gross cap: book scaled to {scale:.1%} of target "
            f"(≤{args.max_gross_mult:g}x {anchor_basis} ${anchor:,.0f})."
        )
    if ai_scale != 1.0:
        ai_eval = res.get("ai_evaluation") or {}
        print(
            f"AI scale: positions scaled to {ai_scale:.1%} "
            f"(confidence={ai_eval.get('confidence')}, reasoning={ai_eval.get('reasoning')!r})."
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
