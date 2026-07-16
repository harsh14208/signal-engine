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
    p.add_argument(
        "--max-gross",
        type=float,
        dest="max_gross",
        default=None,
        help="gross-notional cap (multiple of capital)",
    )
    p.add_argument(
        "--financing-rate",
        type=float,
        dest="financing_rate",
        default=0.0,
        help="annual financing spread on levered gross notional (e.g. 0.01)",
    )
    p.add_argument(
        "--financing-threshold",
        type=float,
        dest="financing_threshold",
        default=1.0,
        help="gross-notional multiple below which no financing is charged",
    )
    p.add_argument(
        "--max-annual-financing-cost",
        type=float,
        dest="max_annual_financing_cost",
        default=None,
        help="hard cap on annual financing cost as a fraction of capital",
    )
    p.add_argument(
        "--challenger",
        action="store_true",
        help="also emit a parallel 'challenger' shadow book with COT flipped, to "
        "forward-test the COT lever without touching the champion book",
    )
    args = p.parse_args(argv)

    overrides = {
        "max_gross_notional": args.max_gross,
        "financing_rate": args.financing_rate,
        "financing_threshold": args.financing_threshold,
        "max_annual_financing_cost": args.max_annual_financing_cost,
    }

    def _emit(book: str, cot: bool, overrides: dict | None = None) -> int:
        try:
            res = generate_target(
                source=args.source,
                cot=cot,
                refresh_cot=not args.no_refresh_cot,
                end=args.end,
                targets_path=args.output,
                book=book,
                overrides=overrides,
            )
        except Exception as exc:
            print(f"generate_targets ({book}) failed: {exc}", file=sys.stderr)
            return 1
        if res.get("skipped"):
            print(f"Target for {res['date']} ({book}) already exists; skipped.")
            return 0
        record = res["record"]
        print(f"Wrote {book} target for {record['date']}: {len(record['units'])} instruments")
        print(
            f"  config: core 19 + governor + {record['buffer_fraction']:.0%} buffer "
            f"+ COT={record['use_cot']}"
        )
        mg = record.get('max_gross_notional')
        fr = record.get('financing_rate', 0.0)
        mg_str = f"max_gross={mg:.1f}x" if mg is not None else "no gross cap"
        print(
            f"  IDM={record['idm']:.2f} FDM={record['fdm']:.2f} governor={record['governor']:.2f} "
            f"{mg_str} financing={fr:.2%}"
        )
        return 0

    champ_cot = not args.no_cot
    rc = _emit("champion", cot=champ_cot, overrides=overrides)
    if args.challenger:
        # Challenger flips the COT lever relative to the champion.
        rc = _emit("challenger", cot=not champ_cot, overrides=overrides) or rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
