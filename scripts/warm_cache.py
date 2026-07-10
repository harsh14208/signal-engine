"""Fetch the live ETF-proxy universe via yfinance → cache to data/prices_universe.parquet.

Run where you have network:
    python scripts/warm_cache.py [START_DATE]
Then run the backtest offline against the cache:
    python -m signal_engine --source cache --validate --oos 0.7
"""

from __future__ import annotations

import sys

from signal_engine.data import load_prices
from signal_engine.markets import symbols


def main(start: str = "2007-01-01") -> int:
    syms = symbols()
    print(f"Fetching {len(syms)} symbols from {start} via yfinance…")
    px = load_prices(syms, start=start, source="yfinance")
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
    raise SystemExit(main(*sys.argv[1:]))
