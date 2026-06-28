"""Daily target-position generator (Tier A1).

Refreshes prices + COT, runs the validated offline config, and emits the next-day
target units as a JSONL record. Designed to run after the US close.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from signal_engine.live import DEFAULT_TARGETS_PATH, generate_target  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Generate today's target positions from the validated config."
    )
    p.add_argument(
        "--source",
        default="auto",
        choices=["auto", "cache", "yfinance", "synthetic"],
        help="price source (auto uses cache, falling back to yfinance)",
    )
    p.add_argument(
        "--no-cot",
        action="store_true",
        dest="no_cot",
        help="disable COT positioning (default: enabled)",
    )
    p.add_argument(
        "--no-refresh-cot",
        action="store_true",
        dest="no_refresh_cot",
        help="use the cached COT panel even if it may be stale",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_TARGETS_PATH,
        help="path to the live-targets JSONL file",
    )
    p.add_argument("--end", default=None, help="optional as-of date (YYYY-MM-DD)")
    args = p.parse_args(argv)

    try:
        res = generate_target(
            source=args.source,
            cot=not args.no_cot,
            refresh_cot=not args.no_refresh_cot,
            end=args.end,
            targets_path=args.output,
        )
    except Exception as exc:
        print(f"generate_targets failed: {exc}", file=sys.stderr)
        return 1

    if res.get("skipped"):
        print(f"Target for {res['date']} already exists; skipped.")
        return 0

    record = res["record"]
    print(f"Wrote target for {record['date']}: {len(record['units'])} instruments")
    print(f"  config: core 19 + governor + {record['buffer_fraction']:.0%} buffer + COT={record['use_cot']}")
    print(f"  IDM={record['idm']:.2f} FDM={record['fdm']:.2f} governor={record['governor']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
