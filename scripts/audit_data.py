"""Phase 2 CLI — audit the price panel for gaps, staleness, jumps, and dead feeds.

Scores the cached/real panel against the NYSE calendar and per-symbol quality
checks, printing a health scorecard and the symbols to quarantine. Exit code 0 if
the panel is healthy, 1 if any symbol is flagged or sessions are missing.

    python scripts/audit_data.py --source cache
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from signal_engine.data import load_prices  # noqa: E402
from signal_engine.data_quality import audit_panel  # noqa: E402
from signal_engine.lineage import lineage_hash, universe_hash  # noqa: E402
from signal_engine.markets import symbols  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Audit the price panel for data quality.")
    p.add_argument("--source", default="cache",
                   choices=["auto", "cache", "yfinance", "synthetic"])
    p.add_argument("--end", default=None, help="as-of date (YYYY-MM-DD)")
    p.add_argument("--min-health", type=float, default=70.0,
                   dest="min_health", help="flag symbols below this health score")
    p.add_argument("--expanded", action="store_true", help="audit the expanded universe")
    args = p.parse_args(argv)

    end = args.end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        prices = load_prices(symbols(args.expanded), start="2007-01-01", end=end,
                             source=args.source, cache_tag="expanded" if args.expanded else "universe")
    except Exception as exc:
        print(f"audit_data failed: {exc}", file=sys.stderr)
        return 1

    rep = audit_panel(prices, end=end, min_health=args.min_health)
    if rep.get("insufficient"):
        print("Insufficient data to audit.", file=sys.stderr)
        return 1

    icon = "✅" if rep["healthy"] else "⚠️"
    print(f"# Data audit — {icon}  lineage={lineage_hash()} universe={universe_hash(args.expanded)}\n")
    print(f"- symbols: {rep['n_symbols']}   mean health: {rep['mean_health']}/100")
    print(f"- panel missing sessions: {rep['panel_missing_sessions']}")
    if rep["panel_missing_dates"]:
        print(f"  first missing: {', '.join(rep['panel_missing_dates'][:10])}")
    if rep["flagged_symbols"]:
        print(f"- ⚠ flagged (<{args.min_health}): {', '.join(rep['flagged_symbols'])}")
        for sym in rep["flagged_symbols"]:
            s = rep["per_symbol"][sym]
            print(f"    {sym:6s} score={s.get('health_score')} gaps={s.get('internal_gaps')} "
                  f"stale={s.get('stale_sessions')} jumps={s.get('jumps')} flat={s.get('max_flatline')}")
    else:
        print("- no symbols flagged")
    return 0 if rep["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
