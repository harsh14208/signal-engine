"""Phase 3 CLI — replay stored decisions and detect engine drift.

Re-derives every point-in-time feature snapshot and classifies divergences.
Exit 0 = clean, 1 = logic drift (code-level; alarming), 3 = error. With
``--enforce`` a logic-drift alarm engages the kill switch, halting order
submission until a human clears it.

    python scripts/detect_drift.py --source cache
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from signal_engine.feature_store import DEFAULT_SNAPSHOT_DIR  # noqa: E402
from signal_engine.live import DEFAULT_KILL_SWITCH_PATH  # noqa: E402
from signal_engine.replay import detect_drift  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Replay-based decision drift detection.")
    p.add_argument("--source", default="cache",
                   choices=["auto", "cache", "yfinance", "synthetic"])
    p.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    p.add_argument("--book", action="append", dest="books",
                   help="restrict to a book (repeatable); default: all")
    p.add_argument("--enforce", action="store_true",
                   help="engage the kill switch on a logic-drift alarm")
    p.add_argument("--kill-switch", type=Path, default=DEFAULT_KILL_SWITCH_PATH)
    args = p.parse_args(argv)

    try:
        rep = detect_drift(snapshot_dir=args.snapshot_dir, source=args.source, books=args.books)
    except Exception as exc:
        print(f"detect_drift failed: {exc}", file=sys.stderr)
        return 3

    if rep["n_snapshots"] == 0:
        print("No snapshots to replay.")
        return 0

    icon = "❌" if rep["alarm"] else "✅"
    print(f"# Decision-drift replay — {icon}")
    print(f"- replayed {rep['n_snapshots']} snapshot(s); matched {rep['matched']}")
    print(f"- by kind: {rep['by_kind']}")
    print(f"- policy changes (non-alarming): {rep['policy_changes']}")
    if rep["unverified"]:
        print(f"- ⚠ unverified: {len(rep['unverified'])} (e.g. {rep['unverified'][0].get('error','')[:80]})")
    for a in rep["alarms"]:
        print(f"  ❌ LOGIC DRIFT {a['as_of']} [{a['book']}]: "
              f"units Δ{a['units_max_diff']:.4g}, forecast Δ{a['forecast_max_diff']:.4g}")

    if rep["alarm"] and args.enforce:
        kill = {
            "paused": True,
            "reason": "logic_drift",
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "details": {"alarms": rep["alarms"][:10]},
        }
        Path(args.kill_switch).write_text(json.dumps(kill, indent=2, default=str))
        print(f"\n⚠ KILL SWITCH ENGAGED (logic_drift) → {args.kill_switch}")

    return 1 if rep["alarm"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
