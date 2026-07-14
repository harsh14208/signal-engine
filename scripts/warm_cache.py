"""Fetch the live ETF-proxy universe via yfinance → cache to data/prices_universe.parquet.

The cache is POINT-IN-TIME: refreshes append new dates (ratio-stitched) and never
rewrite history already traded on; rejected upstream revisions are logged to
data/price_revisions.jsonl. Use --rebase to deliberately accept the fresh
adjusted history wholesale (resets the stitch basis; do this consciously — the
engine's decision history stops being reproducible from the cache).

Run where you have network:
    python scripts/warm_cache.py [--start START_DATE] [--rebase]
Then run the backtest offline against the cache:
    python -m signal_engine --source cache --validate --oos 0.7
"""

from __future__ import annotations

import argparse

from signal_engine.data import load_prices
from signal_engine.markets import symbols


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Warm the PIT price cache from yfinance.")
    p.add_argument("--start", default="2007-01-01", help="history start date")
    p.add_argument(
        "--rebase",
        action="store_true",
        help="discard the cached basis and accept the fresh adjusted history wholesale "
        "(logged to price_revisions.jsonl)",
    )
    # Back-compat: `python scripts/warm_cache.py 2010-01-01` (positional start).
    p.add_argument("start_pos", nargs="?", default=None, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    start = args.start_pos or args.start

    syms = symbols()
    print(f"Fetching {len(syms)} symbols from {start} via yfinance…")
    px = load_prices(syms, start=start, source="yfinance", rebase=args.rebase)
    print(
        f"\nCached {px.shape[1]} instruments × {px.shape[0]} rows → data/prices_universe.parquet\n"
    )
    print(f"{'sym':6s}{'first':12s}{'last':12s}{'bars':>7s}")
    for s in syms:
        if s in px.columns and px[s].first_valid_index() is not None:
            fv, lv = px[s].first_valid_index().date(), px[s].last_valid_index().date()
            print(f"{s:6s}{str(fv):12s}{str(lv):12s}{px[s].notna().sum():>7d}")
        else:
            print(f"{s:6s}{'— MISSING (rate-limited? retry) —':<24s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
