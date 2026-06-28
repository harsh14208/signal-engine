"""No-broker shadow paper book (Tier A2).

Reads the latest target from `data/live_targets.jsonl`, marks the next-day return
using closing prices, and appends it to `data/live_returns.csv`. No broker needed.
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
    append_shadow_return,
    load_latest_target,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Mark the next-day shadow return for the latest target."
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

    target = load_latest_target(args.targets)
    try:
        res = append_shadow_return(
            target=target,
            source=args.source,
            end=args.end,
            returns_path=args.returns,
        )
    except Exception as exc:
        print(f"shadow_book failed: {exc}", file=sys.stderr)
        return 1

    if res.get("skipped"):
        print(f"Skipped: {res.get('reason')} ({res.get('target_date') or res.get('date')})")
        return 0

    record = res["record"]
    print(
        f"Shadow return {record['date']}: {record['live_return']:.4%} "
        f"(mode={record['mode']}, use_cot={record['use_cot']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
