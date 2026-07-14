"""No-broker shadow paper book (Tier A2).

Reads every target from `data/live_targets.jsonl`, marks the next-day return using
closing prices, and appends any not-yet-recorded day to `data/live_returns.csv`.
No broker needed. Marking every target (not just the latest) is deliberate: the
daily loop writes a fresh target *before* this runs, so marking only the latest
would forever hit the one record with no next-day close yet and never accumulate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from signal_engine.live import (  # noqa: E402
    DEFAULT_RETURNS_PATH,
    DEFAULT_TARGETS_PATH,
    mark_all_shadow_returns,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Mark next-day shadow returns for every target (backfilling misses)."
    )
    p.add_argument(
        "--source",
        default="auto",
        choices=["auto", "cache", "yfinance", "synthetic"],
        help="price source for marking",
    )
    p.add_argument(
        "--targets",
        type=Path,
        default=DEFAULT_TARGETS_PATH,
        help="path to live-targets JSONL",
    )
    p.add_argument(
        "--returns",
        type=Path,
        default=DEFAULT_RETURNS_PATH,
        help="path to live-returns CSV",
    )
    p.add_argument(
        "--end",
        default=None,
        help="optional as-of date for historical replay (YYYY-MM-DD)",
    )
    args = p.parse_args(argv)

    try:
        results = mark_all_shadow_returns(
            targets_path=args.targets,
            source=args.source,
            end=args.end,
            returns_path=args.returns,
        )
    except Exception as exc:
        print(f"shadow_book failed: {exc}", file=sys.stderr)
        return 1

    recorded = 0
    for res in results:
        book = res.get("book", "champion")
        if res.get("skipped"):
            # A quiet skip: the newest target has no next-day close yet, or the day
            # is already recorded. Not an error — only worth a note.
            reason = res.get("reason")
            if reason not in ("already_recorded", "no_next_day_price"):
                print(f"Skipped [{book}]: {reason} ({res.get('target_date') or res.get('date')})")
            continue
        record = res["record"]
        recorded += 1
        print(
            f"Shadow return {record['date']} [{book}]: {record['live_return']:.4%} "
            f"(mode={record['mode']}, use_cot={record['use_cot']})"
        )
    if recorded == 0:
        print("No new shadow returns to record (all targets already marked or awaiting close).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
