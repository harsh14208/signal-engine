"""Statistical honesty tooling — ported from the parent project's
backtest_technicals.py and adapted from per-trade to daily-return series.

  • Lo (2002) Sharpe 95% CI         — is SR=0 inside the interval?
  • Deflated Sharpe (Bailey & López de Prado 2014) — can data-mining over
                                       n_trials explain the Sharpe by chance?
  • Block-bootstrap Monte Carlo (Politis & Romano 1994) — honest P5/P95 that
                                       preserve serial autocorrelation.
  • Random-walk placebo             — run the SAME strategy on driftless panels;
                                       a real edge must clear this noise floor.

These are the antidote to the parent project's core mistake: a beautiful edge
that failed its own Deflated Sharpe at the honest trial count. Run them BEFORE
believing any result here.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import ANNUAL_VOL_SQRT, Config
from .data import random_walk_panel  # re-exported: validation.random_walk_panel


def lo_sharpe_ci(daily: pd.Series, n_trials: int = 300) -> dict:
    """Lo (2002) 95% CI on the (annualised) Sharpe + Deflated Sharpe check.

    SE(SR_daily) = sqrt((1 + SR_daily^2 / 2) / N).  Annualised by ×sqrt(256).
    Deflated: expected max SR by chance from n_trials searches (Gumbel EV).
    """
    d = daily.dropna()
    n = len(d)
    if n < 30 or d.std() == 0:
        return {"n": n, "insufficient": True}
    sr_daily = d.mean() / d.std()
    se = math.sqrt((1 + sr_daily**2 / 2) / n)
    lo, hi = sr_daily - 1.96 * se, sr_daily + 1.96 * se
    exp_max_daily = math.sqrt(1.0 / n) * math.sqrt(2 * math.log(max(n_trials, 2)))
    a = ANNUAL_VOL_SQRT
    return {
        "n": n,
        "sharpe": sr_daily * a,
        "ci_low": lo * a,
        "ci_high": hi * a,
        "se": se * a,
        "zero_inside": lo < 0 < hi,
        "n_trials": n_trials,
        "deflated_expected_max": exp_max_daily * a,
        "passes_deflated": (sr_daily * a) > (exp_max_daily * a),
    }


def block_bootstrap_sharpe(daily: pd.Series, n_sims: int = 2000, seed: int = 42) -> dict:
    """Block-bootstrap distribution of the annualised Sharpe.

    Block size = max(5, round(N^(1/3))) preserves autocorrelation; IID resampling
    would understate the CI width by ~30–50%."""
    r = daily.dropna().to_numpy()
    n = len(r)
    if n < 30:
        return {"insufficient": True, "n": n}
    block = max(5, int(round(n ** (1 / 3))))
    n_blocks = -(-n // block)
    rng = np.random.default_rng(seed)
    sims = np.empty(n_sims)
    for i in range(n_sims):
        starts = rng.integers(0, max(1, n - block + 1), size=n_blocks)
        sample = np.concatenate([r[s : s + block] for s in starts])[:n]
        sd = sample.std()
        sims[i] = (sample.mean() / sd * ANNUAL_VOL_SQRT) if sd > 0 else 0.0
    return {
        "n": n,
        "block": block,
        "p5": float(np.percentile(sims, 5)),
        "p50": float(np.percentile(sims, 50)),
        "p95": float(np.percentile(sims, 95)),
        "edge_real": float(np.percentile(sims, 5)) > 0,
    }


def placebo_sharpes(
    strategy_fn,
    n_placebo: int = 20,
    n_instruments: int = 19,
    n_days: int = 3000,
    seed0: int = 100,
) -> dict:
    """Run `strategy_fn(prices)->daily_returns` on `n_placebo` driftless panels.

    Returns the noise-floor distribution of annualised Sharpe. A genuine edge
    must sit clearly ABOVE this band."""
    sharpes = []
    for i in range(n_placebo):
        panel = random_walk_panel(n_instruments, n_days, seed=seed0 + i)
        daily = strategy_fn(panel).dropna()
        if daily.std() > 0:
            sharpes.append(float(daily.mean() / daily.std() * ANNUAL_VOL_SQRT))
    arr = np.array(sharpes) if sharpes else np.array([0.0])
    return {
        "n_placebo": len(sharpes),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "noise_floor_95": float(np.percentile(arr, 95)),
    }


def purged_walk_forward(
    prices: pd.DataFrame,
    config: Config,
    n_splits: int = 5,
    embargo_frac: float = 0.02,
    cot: pd.DataFrame | None = None,
) -> dict:
    """Expanding-window walk-forward with a small embargo gap between train/test.

    For each fold the weights, IDM, and FDM are estimated on the training window
    and then applied out-of-sample on the following test window. The embargo
    prevents target/return overlap leakage.
    """
    from .backtest import run_backtest, run_backtest_with_params
    from .metrics import sharpe

    prices = prices.sort_index().dropna(how="all")
    n = len(prices)
    if n_splits < 2 or n < 300:
        return {"insufficient": True, "n": n}

    boundaries = np.linspace(0, n, n_splits + 1, dtype=int)
    embargo = max(1, int(round(n * embargo_frac)))
    folds = []
    min_train = 128  # enough history for the slowest rules and correlations

    for i in range(1, n_splits):
        train_end = boundaries[i] - embargo
        test_start = boundaries[i]
        test_end = boundaries[i + 1]
        if train_end < min_train or test_end - test_start < 30:
            continue

        train = prices.iloc[:train_end]
        test = prices.iloc[test_start:test_end]
        cot_train = cot.iloc[:train_end] if cot is not None else None
        cot_test = cot.iloc[test_start:test_end] if cot is not None else None

        train_result = run_backtest(train, config, cot=cot_train)
        test_result = run_backtest_with_params(
            test,
            config,
            train_result.weights,
            train_result.idm,
            train_result.fdm,
            cot=cot_test,
        )

        folds.append(
            {
                "train_end": str(train.index[-1]),
                "test_start": str(test.index[0]),
                "test_end": str(test.index[-1]),
                "is_sharpe": sharpe(train_result.daily_returns),
                "oos_sharpe": sharpe(test_result.daily_returns),
            }
        )

    if not folds:
        return {"insufficient": True, "n": n}

    gaps = [f["is_sharpe"] - f["oos_sharpe"] for f in folds]
    return {
        "n_folds": len(folds),
        "folds": folds,
        "mean_is_sharpe": float(np.mean([f["is_sharpe"] for f in folds])),
        "mean_oos_sharpe": float(np.mean([f["oos_sharpe"] for f in folds])),
        "mean_gap": float(np.mean(gaps)),
    }
