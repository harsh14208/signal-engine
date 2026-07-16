"""Daily reconciliation report + guardrails (Tier A3 + A6).

Compares the shadow book's realised daily returns to the engine's modeled returns
for the same config, prints a markdown summary, persists a JSON report, and sets
`data/kill_switch.json` if tracking breaks down or the rolling edge decays.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from signal_engine.live import (  # noqa: E402
    DEFAULT_KILL_SWITCH_PATH,
    DEFAULT_RECON_DIR,
    DEFAULT_RETURNS_PATH,
    DEFAULT_TARGETS_PATH,
    load_latest_target_for_book,
    run_reconciliation,
)


def _fmt_report(report: dict) -> str:
    rec = report["reconciliation"]
    edge = report["edge_decay"]
    lines = ["# Forward reconciliation report\n"]
    lines.append(f"- Report date: **{report['date']}**")
    lines.append(f"- Target date: **{report['target_date']}**")
    lines.append(
        f"- Comparison: **modeled {report.get('compare_to', 'net')}** vs live {report['target_date']}"
    )
    lines.append(f"- Modeled Sharpe (full history): **{report['modeled_sharpe']:.2f}**")
    if report.get("live_sharpe") is not None:
        lines.append(f"- Live Sharpe (to date): **{report['live_sharpe']:.2f}**")
    lines.append("")
    lines.append("## Live vs backtest\n")
    if rec.get("insufficient"):
        lines.append(f"- Insufficient live data (n={rec.get('n', 0)}).")
    else:
        status = "✅ aligned" if rec["aligned"] else "⚠ NOT aligned"
        lines.append(
            f"- n={rec['n']}  corr={rec['corr']:.2f}  "
            f"tracking_error={rec['tracking_error']:.2%}  drift={rec['drift']:.2%}  **{status}**"
        )
    lines.append("")
    lines.append("## Edge decay (rolling 1y Sharpe)\n")
    if edge.get("insufficient"):
        lines.append("- Insufficient live data.")
    else:
        alarm = "⚠ ALARM" if edge["alarm"] else "✅ healthy"
        wq = edge.get("worst_quartile")
        wq_note = f"  worst_quartile={'yes' if wq else 'no'}" if wq is not None else ""
        lines.append(
            f"- current **{edge['current']:.2f}**  median {edge['median']:.2f}  "
            f"min {edge['min']:.2f}  max {edge['max']:.2f}  "
            f"% windows <0: {edge['pct_windows_below_zero']:.0%}{wq_note}  ({alarm})"
        )
    decomp = rec.get("drift_decomposition") or {}
    if not decomp.get("insufficient"):
        lines.append("")
        lines.append("## Drift decomposition (implementation shortfall)\n")
        lines.append(
            f"- total drift {decomp.get('total_drift', 0):.2%}: "
            f"alpha {decomp.get('alpha', 0):.2%} + beta_gap {decomp.get('beta_gap', 0):.2%}"
        )
        lines.append(
            f"- beta={decomp.get('beta', 0):.2f}  residual={decomp.get('residual', 0):.2%}"
        )
        if "delay" in decomp:
            lines.append(
                f"- Perold components: delay {decomp['delay']:.2%}, "
                f"opportunity {decomp['opportunity']:.2%}, "
                f"execution residual {decomp['execution_residual']:.2%}"
            )
    rev = report.get("input_revision")
    if rev is not None:
        lines.append("")
        lines.append("## Input revision (stored vs recomputed forecasts)\n")
        if rev.get("clean"):
            lines.append(
                f"- ✅ clean — {rev.get('n_targets_checked', 0)} recent target(s) "
                "reproduce from today's data"
            )
        else:
            lines.append(
                f"- ⚠ INPUTS REVISED under {rev['n_symbols_revised']} symbol(s) "
                f"(checked {rev['n_targets_checked']} targets):"
            )
            for sym, d in sorted(
                rev["revised"].items(), key=lambda kv: -kv[1]["max_abs_diff"]
            ):
                lines.append(
                    f"  - **{sym}** {d['date']}: stored {d['stored']:+.1f} → "
                    f"recomputed {d['recomputed']:+.1f} (Δ{d['max_abs_diff']:.1f})"
                )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Reconcile shadow/live returns against the backtest.")
    p.add_argument(
        "--source",
        default="auto",
        choices=["auto", "cache", "yfinance", "synthetic"],
        help="price source for the modeled backtest",
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
        "--recon-dir",
        type=Path,
        default=DEFAULT_RECON_DIR,
        help="directory for daily reconciliation JSON reports",
    )
    p.add_argument(
        "--kill-switch",
        type=Path,
        default=DEFAULT_KILL_SWITCH_PATH,
        help="path to the kill-switch JSON file",
    )
    p.add_argument(
        "--alarm-floor",
        type=float,
        default=0.0,
        help="rolling-Sharpe floor that trips the edge-decay alarm",
    )
    p.add_argument(
        "--alarm-on-worst-quartile",
        action="store_true",
        dest="alarm_on_worst_quartile",
        help="also trip the edge-decay alarm when rolling Sharpe falls into its "
        "own worst quartile (self-calibrating decay detector)",
    )
    p.add_argument(
        "--book",
        default="champion",
        help="shadow book to reconcile (default: champion — the deployed book; "
        "NOT simply the last target written, which may be a challenger)",
    )
    args = p.parse_args(argv)

    target = load_latest_target_for_book(args.targets, args.book)
    try:
        res = run_reconciliation(
            target=target,
            source=args.source,
            returns_path=args.returns,
            recon_dir=args.recon_dir,
            kill_switch_path=args.kill_switch,
            alarm_floor=args.alarm_floor,
            alarm_on_worst_quartile=args.alarm_on_worst_quartile,
            targets_path=args.targets,
        )
    except Exception as exc:
        print(f"reconcile failed: {exc}", file=sys.stderr)
        return 1

    print(_fmt_report(res["report"]))
    print(f"\nReport written to: {res['path']}")
    if res["kill_switch"]["paused"]:
        print(
            f"\n⚠ KILL SWITCH ENGAGED ({res['kill_switch']['reason']}). "
            "Review before executing new orders."
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
