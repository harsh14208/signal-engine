"""Phase 0 — the go/no-go edge gate.

One command that runs the full pre-registered honesty battery on REAL data and
returns a verdict: does the edge survive, and does it survive *costs*?  This is
the gate that decides whether any platform investment is justified — the check
the parent engine never had before stacking 96 sections on a dead edge.

Pre-registered criteria (fixed before looking at the result):

  HARD gates (any failure ⇒ FAIL — the edge is indistinguishable from luck):
    H1 clears_noise      net Sharpe > placebo 95th-percentile noise floor
    H2 edge_real         block-bootstrap 5th-percentile Sharpe > 0
    H3 passes_deflated   Deflated Sharpe passes at the honest trial count

  ROBUSTNESS gates (hard pass but any failure ⇒ CONDITIONAL, not PASS):
    R1 cpcv_robust       CPCV OOS 5th pct > 0 and < 25% of paths below zero
    R2 walk_forward_ok   walk-forward mean OOS Sharpe > 0 and IS−OOS gap < 0.5
    R3 cost_headroom     break-even cost ≥ 2× the assumed cost (Phase 1 tie-in)

ENB / diversification and the forward live track are reported as context, not
gates (ENB is a design check; the live track needs accrued days to be decisive).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .backtest import run_backtest
from .config import Config
from .data_quality import audit_panel
from .diagnostics import diversification_audit
from .friction import cost_break_even
from .metrics import sharpe
from .monitor import edge_decay_report
from .validation import (
    block_bootstrap_sharpe,
    combinatorial_purged_cv,
    honest_n_trials,
    lo_sharpe_ci,
    placebo_sharpes,
    purged_walk_forward,
)

# Conservative prior trial count when the registry is empty — matches the count
# used in experiment_results.md so an unpopulated registry can't game deflation.
_TRIAL_FLOOR = 100


def _placebo_config(config: Config) -> Config:
    """A cot/carry-free clone so placebo panels (pure noise, no external data) run."""
    from dataclasses import replace

    return replace(
        config,
        use_cot=False,
        use_carry=False,
        use_carry_proxies=False,
        use_network_momentum=False,
    )


def evaluate_edge(
    prices: pd.DataFrame,
    config: Config | None = None,
    cot: pd.DataFrame | None = None,
    live_returns: pd.Series | None = None,
    n_trials: int | None = None,
) -> dict[str, Any]:
    """Run the battery and return metrics, gate booleans, and an overall verdict."""
    config = config or Config()
    result = run_backtest(prices, config, cot=cot)
    daily = result.daily_returns
    net_sr = sharpe(daily)

    # Trial count for deflation: honest registry count, floored conservatively.
    trials = n_trials if n_trials is not None else max(honest_n_trials(), _TRIAL_FLOOR)

    lo = lo_sharpe_ci(daily, n_trials=trials)
    boot = block_bootstrap_sharpe(daily)
    wf = purged_walk_forward(prices, config, cot=cot)
    cpcv = combinatorial_purged_cv(prices, config, n_groups=6, k_test=2, cot=cot)

    pcfg = _placebo_config(config)
    placebo = placebo_sharpes(lambda p: run_backtest(p, pcfg).daily_returns,
                              n_instruments=min(len(prices.columns), 19))
    breakeven = cost_break_even(prices, config, cot=cot)
    divers = diversification_audit(result)
    data_health = audit_panel(prices)

    # ── Gates ────────────────────────────────────────────────────────────────
    noise_floor = placebo.get("noise_floor_95", 0.0)
    h1 = net_sr > noise_floor
    h2 = bool(boot.get("edge_real")) if not boot.get("insufficient") else False
    h3 = bool(lo.get("passes_deflated")) if not lo.get("insufficient") else False

    r1 = (
        not cpcv.get("insufficient")
        and cpcv.get("oos_p5", -1) > 0
        and cpcv.get("pct_paths_below_zero", 1.0) < 0.25
    )
    r2 = (
        not wf.get("insufficient")
        and wf.get("mean_oos_sharpe", -1) > 0
        and wf.get("mean_gap", 99) < 0.5
    )
    be = breakeven.get("break_even_bps")
    headroom = breakeven.get("headroom_x")
    r3 = be is None or (headroom is not None and headroom >= 2.0)

    hard = {"H1_clears_noise": h1, "H2_edge_real": h2, "H3_passes_deflated": h3}
    robustness = {"R1_cpcv_robust": r1, "R2_walk_forward_ok": r2, "R3_cost_headroom": r3}

    if not all(hard.values()):
        verdict = "FAIL"
    elif all(robustness.values()):
        verdict = "PASS"
    else:
        verdict = "CONDITIONAL"

    forward = None
    if live_returns is not None and len(live_returns.dropna()) >= 20:
        lr = live_returns.dropna()
        forward = {
            "n_days": int(len(lr)),
            "live_sharpe": float(sharpe(lr)),
            "edge_decay": edge_decay_report(lr),
        }

    return {
        "verdict": verdict,
        "hard_gates": hard,
        "robustness_gates": robustness,
        "metrics": {
            "net_sharpe": net_sr,
            "noise_floor_95": noise_floor,
            "bootstrap_p5": boot.get("p5"),
            "deflated_expected_max": lo.get("deflated_expected_max"),
            "n_trials": trials,
            "cpcv_oos_p5": cpcv.get("oos_p5"),
            "cpcv_pct_below_zero": cpcv.get("pct_paths_below_zero"),
            "wf_mean_oos_sharpe": wf.get("mean_oos_sharpe"),
            "wf_mean_gap": wf.get("mean_gap"),
            "break_even_bps": be,
            "cost_headroom_x": headroom,
            "effective_bets": divers.get("effective_bets"),
            "idm_vs_effective": divers.get("idm_vs_effective"),
            "data_mean_health": data_health.get("mean_health"),
            "data_flagged": data_health.get("flagged_symbols"),
        },
        "forward_track": forward,
        "raw": {
            "placebo": placebo,
            "bootstrap": boot,
            "deflated": lo,
            "walk_forward": wf,
            "cpcv": cpcv,
            "break_even": {k: v for k, v in breakeven.items() if k != "curve"},
            "diversification": divers,
        },
    }


def promotion_decision(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
    forward_report: dict[str, Any] | None = None,
    candidate_book: str | None = None,
    comparison_pbo: float | None = None,
    **paired_kwargs,
) -> dict[str, Any]:
    """Return a structured promotion verdict for a candidate vs baseline.

    A lever ships as default only if one of two conditions holds:
      (a) paired walk-forward fold delta vs baseline has a 95% CI that excludes 0;
      (b) it has WON the forward challenger gate — `champion_challenger_report`
          recommends promoting this candidate's book, which requires ≥ its
          `min_days` (default 60) of forward data AND beating the champion's
          Sharpe by the promote margin. Days accrued alone are not evidence:
          a challenger that merely *survives* 60 days may have lost throughout.

    If the comparison set's PBO > 0.5, backtest-only promotion is blocked outright
    (the search itself is overfit).  This function captures the discipline that
    today lives in chat transcripts.
    """
    baseline_wf = baseline_report.get("raw", {}).get("walk_forward", {})
    candidate_wf = candidate_report.get("raw", {}).get("walk_forward", {})
    paired = None
    if baseline_wf and candidate_wf and not baseline_wf.get("insufficient"):
        from .validation import paired_fold_comparison

        paired = paired_fold_comparison(
            baseline_wf.get("folds", []), candidate_wf.get("folds", []), **paired_kwargs
        )

    # Forward evidence = the challenger gate's own verdict, not raw days accrued.
    forward_won = bool(
        forward_report is not None
        and candidate_book
        and forward_report.get("recommendation") == f"promote:{candidate_book}"
    )
    forward_days = 0
    if forward_report is not None and candidate_book:
        forward_days = int(
            (forward_report.get("books", {}).get(candidate_book) or {}).get("n", 0)
        )

    backtest_promotable = (
        paired is not None
        and not paired.get("indistinguishable", True)
        and paired.get("mean_delta", 0.0) > 0.0
    )
    if comparison_pbo is not None and comparison_pbo > 0.5:
        backtest_promotable = False

    verdict = "PROMOTE" if (backtest_promotable or forward_won) else "HOLD"
    reason = []
    if backtest_promotable:
        reason.append("paired fold delta CI excludes 0 and favors candidate")
    if forward_won:
        reason.append(
            f"forward challenger gate WON ({candidate_book}, n={forward_days} days, "
            "beat champion by the promote margin)"
        )
    if comparison_pbo is not None and comparison_pbo > 0.5:
        reason.append(f"PBO {comparison_pbo:.2f} > 0.5 blocks backtest-only promotion")
    if not reason:
        reason.append("no promotion criterion met")

    return {
        "verdict": verdict,
        "forward_days": forward_days,
        "comparison_pbo": comparison_pbo,
        "paired_fold_comparison": paired,
        "backtest_promotable": backtest_promotable,
        "forward_won": forward_won,
        "reason": "; ".join(reason),
    }


def format_verdict(report: dict[str, Any]) -> str:
    """Human-readable gate report for the CLI."""
    m = report["metrics"]
    icon = {"PASS": "✅", "CONDITIONAL": "⚠️", "FAIL": "❌"}[report["verdict"]]
    lines = [f"# Edge gate — {icon} {report['verdict']}\n"]

    def _row(name: str, ok: bool, detail: str) -> str:
        return f"  {'✅' if ok else '❌'} {name}: {detail}"

    lines.append("## Hard gates (any fail ⇒ FAIL)")
    lines.append(_row("H1 clears_noise", report["hard_gates"]["H1_clears_noise"],
                      f"net {m['net_sharpe']:.2f} vs noise-floor {m['noise_floor_95']:.2f}"))
    lines.append(_row("H2 edge_real", report["hard_gates"]["H2_edge_real"],
                      f"bootstrap P5 {m['bootstrap_p5']:.2f} > 0"))
    lines.append(_row("H3 passes_deflated", report["hard_gates"]["H3_passes_deflated"],
                      f"net {m['net_sharpe']:.2f} vs deflated-max {m['deflated_expected_max']:.2f} "
                      f"(n_trials={m['n_trials']})"))

    lines.append("\n## Robustness gates (fail ⇒ CONDITIONAL)")
    lines.append(_row("R1 cpcv_robust", report["robustness_gates"]["R1_cpcv_robust"],
                      f"OOS P5 {m['cpcv_oos_p5']:.2f}, {m['cpcv_pct_below_zero']:.0%} paths<0"))
    lines.append(_row("R2 walk_forward_ok", report["robustness_gates"]["R2_walk_forward_ok"],
                      f"OOS {m['wf_mean_oos_sharpe']:.2f}, gap {m['wf_mean_gap']:.2f}"))
    be = m["break_even_bps"]
    be_s = "none≤grid" if be is None else f"{be:.1f}bps ({m['cost_headroom_x']:.1f}×)"
    lines.append(_row("R3 cost_headroom", report["robustness_gates"]["R3_cost_headroom"],
                      f"break-even {be_s}"))

    lines.append("\n## Context (not gates)")
    lines.append(f"  • effective bets {m['effective_bets']:.1f}  (IDM/ENB {m['idm_vs_effective']:.2f})")
    dh = m.get("data_mean_health")
    if dh is not None:
        flagged = m.get("data_flagged") or []
        flag_s = f", flagged: {', '.join(flagged)}" if flagged else ""
        lines.append(f"  • data health {dh:.0f}/100{flag_s}")
    if report["forward_track"]:
        ft = report["forward_track"]
        lines.append(f"  • forward: n={ft['n_days']} live Sharpe {ft['live_sharpe']:.2f}")
    else:
        lines.append("  • forward: no live track yet (needs ≥20 accrued days)")
    return "\n".join(lines)
