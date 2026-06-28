from signal_engine.backtest import run_backtest
from signal_engine.config import Config
from signal_engine.data import synthetic_prices
from signal_engine.markets import symbols
from signal_engine.report import diversification_report, full_report, headline_report


def test_headline_report_runs(result):
    text = headline_report(result)
    assert "Headline" in text
    assert "Sharpe" in text


def test_diversification_report_runs(result):
    text = diversification_report(result)
    assert "Diversification" in text
    assert "Portfolio Sharpe" in text


def test_full_report_runs(result):
    text = full_report(result)
    assert "Headline" in text
    assert "Diversification" in text


def test_diversification_report_handles_expanded_universe():
    """Symbols only in the expanded universe must not crash the class lookup."""
    expanded_syms = [s for s in symbols(expanded=True) if s not in symbols(expanded=False)][:5]
    px = synthetic_prices(expanded_syms, n_days=900, seed=7)
    cfg = Config(use_expanded_universe=True, weight_scheme="cluster")
    res = run_backtest(px, cfg)
    text = diversification_report(res)
    assert "Portfolio Sharpe" in text
    # Every symbol should have a non-empty class column.
    for line in text.splitlines():
        if line.startswith("|") and "Instrument" not in line and "|:" not in line:
            parts = [p.strip() for p in line.split("|")]
            # parts[0] empty, [1] sym, [2] class, [3] sharpe.
            assert len(parts) >= 4
            assert parts[2] and parts[2] != "—"
