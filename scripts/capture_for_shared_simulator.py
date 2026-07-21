"""Capture raw per-instrument signal-engine output for a shared simulator.

Runs the signal-engine backtest on cached prices (or yfinance if cache is
missing) and writes the per-instrument arrays that the downstream
portfolio-level combination step needs:

    - prices
    - combined forecasts
    - per-instrument net/gross returns
    - per-instrument positions (buffered, shifted = effective units)
    - per-instrument notional
    - portfolio daily returns / equity

Usage from the signal-engine checkout:
    .venv/bin/python scripts/capture_for_shared_simulator.py \
        --output-dir /Users/harshv.singh/TradingRecommendationSystem/analysis_output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Import signal_engine from the parent checkout.
HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from signal_engine.backtest import run_backtest  # noqa: E402
from signal_engine.config import Config  # noqa: E402
from signal_engine.data import load_prices  # noqa: E402
from signal_engine.carry_data import build_carry_panel  # noqa: E402
from signal_engine.markets import symbols as _market_symbols  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture signal-engine per-instrument data")
    parser.add_argument("--source", default="cache", choices=["synthetic", "yfinance", "cache", "auto"])
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tag", default="signal_engine", help="prefix for output files")
    # Re-use the same flag taxonomy as the CLI for reproducibility.
    parser.add_argument("--vol-target", type=float, default=0.20)
    parser.add_argument("--cost-bps", type=float, default=2.5)
    parser.add_argument("--buffer", type=float, default=0.10)
    parser.add_argument("--no-governor", action="store_true")
    parser.add_argument("--no-breakout", action="store_true")
    parser.add_argument("--carry", action="store_true")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--network-momentum", action="store_true")
    parser.add_argument("--semis", action="store_true")
    parser.add_argument("--qqq", action="store_true")
    parser.add_argument("--expanded-universe", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.end = args.end or None
    args.start = args.start or None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    syms = _market_symbols(expanded=args.expanded_universe)
    # Add optional extensions requested by CLI flags.
    if args.semis:
        syms = list(set(syms + ["SMH", "SOXX", "XSD"]))
    if args.qqq:
        syms = list(set(syms + ["QQQ"]))

    print(f"Loading prices (source={args.source}) ...")
    prices = load_prices(
        syms,
        source=args.source,
        start=args.start,
        end=args.end,
    )
    print(f"  shape={prices.shape}, dates={prices.index[0].date()} to {prices.index[-1].date()}")

    cfg = Config(
        vol_target=args.vol_target,
        cost_bps=args.cost_bps,
        buffer_fraction=args.buffer,
        use_governor=not args.no_governor,
        use_breakout=not args.no_breakout,
        use_carry=args.carry,
        use_cot=args.cot,
        use_network_momentum=args.network_momentum,
    )

    carry = build_carry_panel(prices) if args.carry else None

    print("Running backtest ...")
    result = run_backtest(prices, config=cfg, carry=carry)
    print(f"  idm={result.idm:.3f}, fdm={result.fdm:.3f}, instruments={len(result.weights)}")

    tag = args.tag
    # Write CSVs with the arrays the downstream simulator needs.
    prices.to_csv(out_dir / f"{tag}_prices.csv")
    result.forecasts.to_csv(out_dir / f"{tag}_forecasts.csv")
    result.per_instrument_returns.to_csv(out_dir / f"{tag}_per_instrument_returns.csv")
    result.per_instrument_gross.to_csv(out_dir / f"{tag}_per_instrument_gross.csv")
    result.positions.to_csv(out_dir / f"{tag}_positions.csv")
    result.buffered.to_csv(out_dir / f"{tag}_buffered.csv")
    result.notional.to_csv(out_dir / f"{tag}_notional.csv")
    result.daily_returns.to_frame("daily_return").to_csv(out_dir / f"{tag}_daily_returns.csv")
    result.equity.to_frame("equity").to_csv(out_dir / f"{tag}_equity.csv")
    result.turnover.to_frame("turnover").to_csv(out_dir / f"{tag}_turnover.csv")
    result.governor.to_frame("governor").to_csv(out_dir / f"{tag}_governor.csv")
    result.overlay.to_frame("overlay").to_csv(out_dir / f"{tag}_overlay.csv")

    metadata = {
        "idm": float(result.idm),
        "fdm": float(result.fdm),
        "weights": {k: float(v) for k, v in result.weights.items()},
        "config": {
            "vol_target": float(cfg.vol_target),
            "cost_bps": float(cfg.cost_bps),
            "buffer_fraction": float(cfg.buffer_fraction),
            "use_governor": bool(cfg.use_governor),
            "use_breakout": bool(cfg.use_breakout),
            "use_carry": bool(cfg.use_carry),
            "use_cot": bool(cfg.use_cot),
            "use_network_momentum": bool(cfg.use_network_momentum),
        },
    }
    (out_dir / f"{tag}_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Wrote {tag}* files to {out_dir}")


if __name__ == "__main__":
    main()
