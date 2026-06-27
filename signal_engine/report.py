"""Human-readable reports — including the one that matters most: the
*diversification* report, which makes the entire edge thesis visible.

If the engine works, you will see: mean standalone instrument Sharpe ≈ 0.2–0.4,
mean pairwise correlation low, and a PORTFOLIO Sharpe several times higher. That
gap is the free lunch — and it is exactly what a single over-fit equity strategy
can never produce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import BacktestResult
from .markets import BY_SYMBOL
from .metrics import sharpe, summary


def _avg_offdiag_corr(df: pd.DataFrame) -> float:
    c = df.corr()
    n = c.shape[0]
    if n < 2:
        return float("nan")
    return float(np.nanmean(c.values[np.triu_indices(n, 1)]))


def _fmt_pct(x: float) -> str:
    return "—" if x is None or np.isnan(x) else f"{x:+.1%}"


def headline_report(result: BacktestResult) -> str:
    s = summary(result.equity, result.daily_returns, result.turnover)
    g = summary(result.equity, result.gross_returns)
    lines = [
        "## Headline (net of costs)",
        "",
        f"- Config: {result.config.describe()}",
        f"- IDM: {result.idm:.2f}   FDM: {result.fdm:.2f}   instruments: "
        f"{result.per_instrument_returns.shape[1]}   days: {s['n_days']}",
        "",
        "| Metric | Net | Gross |",
        "|:--|--:|--:|",
        f"| Sharpe | {s['sharpe']:.2f} | {g['sharpe']:.2f} |",
        f"| Ann. return | {_fmt_pct(s['ann_return'])} | {_fmt_pct(g['ann_return'])} |",
        f"| Ann. vol | {s['ann_vol']:.1%} | {g['ann_vol']:.1%} |",
        f"| Max drawdown | {_fmt_pct(s['max_drawdown'])} | — |",
        f"| CAGR | {_fmt_pct(s['cagr'])} | — |",
        f"| Calmar | {s['calmar']:.2f} | — |",
        f"| Sortino | {s['sortino']:.2f} | — |",
        f"| Skew | {s['skew']:+.2f} | — |",
        f"| Ann. turnover | {s.get('ann_turnover', float('nan')):.1f}x | — |",
    ]
    return "\n".join(lines)


def diversification_report(result: BacktestResult) -> str:
    pir = result.per_instrument_returns
    standalone = {c: sharpe(pir[c]) for c in pir.columns}
    mean_standalone = float(np.nanmean(list(standalone.values())))
    avg_corr = _avg_offdiag_corr(pir)
    port_sharpe = sharpe(result.daily_returns)
    ratio = port_sharpe / mean_standalone if mean_standalone not in (0, np.nan) else float("nan")

    rows = sorted(standalone.items(), key=lambda kv: kv[1] if not np.isnan(kv[1]) else -9)
    table = ["| Instrument | Class | Standalone Sharpe |", "|:--|:--|--:|"]
    for sym, sr in reversed(rows):
        cls = BY_SYMBOL[sym].asset_class if sym in BY_SYMBOL else "—"
        table.append(f"| {sym} | {cls} | {sr:.2f} |")

    lines = [
        "## Diversification — the edge made visible",
        "",
        f"- Mean **standalone** instrument Sharpe: **{mean_standalone:.2f}**  "
        "(no single bet is impressive — by design)",
        f"- Mean pairwise correlation of instrument strategies: **{avg_corr:.2f}**",
        f"- IDM (1/√w'Ρw): **{result.idm:.2f}**",
        f"- **Portfolio Sharpe: {port_sharpe:.2f}**",
        f"- **Diversification ratio (portfolio / mean standalone): "
        f"{ratio:.1f}×**  ← this is the whole thesis",
        "",
        *table,
    ]
    return "\n".join(lines)


def full_report(result: BacktestResult) -> str:
    return headline_report(result) + "\n\n" + diversification_report(result)
