"""Command line: load data → backtest → report → (optional) validate.

python -m signal_engine                      # synthetic demo (offline)
python -m signal_engine --validate           # + Lo CI, Deflated SR, MC, placebo
python -m signal_engine --source yfinance     # real ETF-proxy data (needs network)
python -m signal_engine --oos 0.7 --validate  # chronological IS/OOS split
"""

from __future__ import annotations

import argparse

import numpy as np

from .backtest import run_backtest
from .config import Config
from .data import load_prices, synthetic_carry
from .markets import symbols
from .metrics import sharpe
from .report import full_report
from .validation import block_bootstrap_sharpe, lo_sharpe_ci, placebo_sharpes


def build_config(args) -> Config:
    return Config(
        capital=args.capital,
        vol_target=args.vol_target,
        use_breakout=not args.no_breakout,
        use_carry=args.carry,
        cost_bps=args.cost_bps,
        buffer_fraction=args.buffer,
    )


def _print_validation(result, cfg, args) -> None:
    print("\n## Statistical validation\n")
    lo = lo_sharpe_ci(result.daily_returns, n_trials=args.n_trials)
    if lo.get("insufficient"):
        print("- Insufficient data for Lo CI.")
    else:
        verdict = "✅ outside" if not lo["zero_inside"] else "⚠ INSIDE"
        print(
            f"- **Lo (2002) 95% CI (annualised):** [{lo['ci_low']:.2f}, {lo['ci_high']:.2f}] "
            f"— SR=0 is {verdict} (N={lo['n']} days)"
        )
        dverdict = "✅ clears" if lo["passes_deflated"] else "⚠ FAILS"
        print(
            f"- **Deflated Sharpe ({lo['n_trials']} trials):** expected max by chance = "
            f"{lo['deflated_expected_max']:.2f} → Sharpe {lo['sharpe']:.2f} {dverdict} it"
        )

    mc = block_bootstrap_sharpe(result.daily_returns)
    if not mc.get("insufficient"):
        edge = "✅ > 0" if mc["edge_real"] else "⚠ ≤ 0"
        print(
            f"- **Block-bootstrap MC (block={mc['block']}):** P5={mc['p5']:.2f} / "
            f"P50={mc['p50']:.2f} / P95={mc['p95']:.2f} — P5 {edge}"
        )

    # Random-walk placebo: same strategy on driftless panels = noise floor.
    n_days = min(len(result.daily_returns), 2500)
    pl = placebo_sharpes(
        lambda panel: run_backtest(panel, cfg).daily_returns,
        n_placebo=args.placebo,
        n_instruments=result.per_instrument_returns.shape[1],
        n_days=n_days,
    )
    port = sharpe(result.daily_returns)
    clears = "✅ clears" if port > pl["noise_floor_95"] else "⚠ does NOT clear"
    print(
        f"- **Random-walk placebo (n={pl['n_placebo']}):** noise floor "
        f"mean={pl['mean']:.2f}, 95th pct={pl['noise_floor_95']:.2f} → "
        f"real Sharpe {port:.2f} {clears} the floor"
    )


def _print_oos(result, frac: float) -> None:
    d = result.daily_returns.dropna()
    split = int(len(d) * frac)
    is_sr, oos_sr = sharpe(d.iloc[:split]), sharpe(d.iloc[split:])
    print("\n## Out-of-sample (chronological split)\n")
    print(f"- IS Sharpe (first {frac:.0%}): **{is_sr:.2f}**")
    print(f"- OOS Sharpe (last {1 - frac:.0%}): **{oos_sr:.2f}**")
    gap = is_sr - oos_sr
    flag = "✅ small" if abs(gap) < 0.2 else "⚠ large"
    print(f"- Curation gap (IS − OOS): **{gap:+.2f}** ({flag})")


def run(args) -> int:
    cfg = build_config(args)
    syms = symbols()
    prices = load_prices(syms, start=args.start, end=args.end, source=args.source)
    if prices.shape[1] < 2:
        print("Not enough instruments with data.")
        return 1

    carry = None
    if args.carry:
        # Demo carry only. Real carry needs term-structure data — see README §carry.
        carry = synthetic_carry(list(prices.columns), prices.index)

    result = run_backtest(prices, cfg, carry=carry)
    print(f"# signal-engine — {args.source} run\n")
    print(full_report(result))
    if args.oos:
        _print_oos(result, args.oos)
    if args.validate:
        _print_validation(result, cfg, args)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="signal_engine", description=__doc__)
    p.add_argument(
        "--source", default="synthetic", choices=["synthetic", "yfinance", "cache", "auto"]
    )
    p.add_argument("--start", default="2007-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--capital", type=float, default=1_000_000.0)
    p.add_argument("--vol-target", type=float, default=0.20, dest="vol_target")
    p.add_argument("--cost-bps", type=float, default=1.5, dest="cost_bps")
    p.add_argument("--buffer", type=float, default=0.10)
    p.add_argument("--no-breakout", action="store_true")
    p.add_argument("--carry", action="store_true", help="demo carry rule (synthetic series)")
    p.add_argument("--validate", action="store_true", help="run the statistical honesty suite")
    p.add_argument(
        "--oos",
        type=float,
        default=None,
        metavar="FRAC",
        help="chronological IS/OOS split, e.g. 0.7",
    )
    p.add_argument(
        "--n-trials", type=int, default=100, dest="n_trials", help="Deflated-Sharpe trial count"
    )
    p.add_argument("--placebo", type=int, default=12, help="random-walk placebo runs")
    args = p.parse_args(argv)
    np.seterr(all="ignore")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
