"""Phase 0 CLI — run the edge gate on real data and emit a go/no-go verdict.

Loads the cached/real price panel, builds the validated forward config, runs the
full pre-registered battery (placebo, bootstrap, deflated Sharpe, walk-forward,
CPCV, cost break-even, diversification) and — if a live-returns file exists — the
forward track. Exit code: 0 = PASS, 1 = CONDITIONAL, 2 = FAIL, 3 = error.

    python scripts/validate_edge.py --source cache
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from signal_engine.cot_data import build_cot_forecast_panel  # noqa: E402
from signal_engine.data import load_prices  # noqa: E402
from signal_engine.edge_gate import evaluate_edge, format_verdict  # noqa: E402
from signal_engine.live import DEFAULT_RETURNS_PATH, load_live_returns, validated_config  # noqa: E402
from signal_engine.markets import symbols  # noqa: E402
from signal_engine.validation import register_trial  # noqa: E402

_EXIT = {"PASS": 0, "CONDITIONAL": 1, "FAIL": 2}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 0 edge gate: does the edge survive?")
    p.add_argument("--source", default="cache",
                   choices=["auto", "cache", "yfinance", "synthetic"],
                   help="price source (default: cache — use real data, not synthetic)")
    p.add_argument("--no-cot", action="store_true", dest="no_cot")
    p.add_argument("--end", default=None, help="optional as-of date (YYYY-MM-DD)")
    p.add_argument("--returns", type=Path, default=DEFAULT_RETURNS_PATH,
                   help="live-returns CSV for the forward-track readout")
    p.add_argument("--n-trials", type=int, default=None,
                   help="override deflation trial count (default: honest registry count)")
    args = p.parse_args(argv)

    if args.source == "synthetic":
        print("⚠️  --source synthetic is a smoke test only; the gate is meaningful on real data.")

    cfg = validated_config(cot=not args.no_cot)
    register_trial(cfg, label="edge_gate")  # this evaluation counts as a trial

    end = args.end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        prices = load_prices(symbols(), start="2007-01-01", end=end,
                             source=args.source, cache_tag="universe")
        prices = prices.loc[prices.index <= end] if args.end else prices
        if prices.empty or len(prices) < 300:
            print(f"Insufficient price history ({len(prices)} rows).", file=sys.stderr)
            return 3
        cot = build_cot_forecast_panel(prices, tag="core") if cfg.use_cot else None
        live = load_live_returns(args.returns) if Path(args.returns).exists() else None
        report = evaluate_edge(prices, cfg, cot=cot, live_returns=live, n_trials=args.n_trials)
    except Exception as exc:
        print(f"validate_edge failed: {exc}", file=sys.stderr)
        return 3

    print(format_verdict(report))
    return _EXIT[report["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
