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

import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ANNUAL_VOL_SQRT, Config
from .data import random_walk_panel  # re-exported: validation.random_walk_panel

_DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_TRIAL_REGISTRY = _DATA_DIR / "trial_registry.jsonl"


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


def combinatorial_purged_cv(
    prices: pd.DataFrame,
    config: Config,
    n_groups: int = 6,
    k_test: int = 2,
    embargo_frac: float = 0.02,
    cot: pd.DataFrame | None = None,
) -> dict:
    """Combinatorial Purged Cross-Validation (López de Prado) for one strategy.

    The timeline is partitioned into ``n_groups`` contiguous blocks.  For every
    combination that holds out ``k_test`` blocks as the test set, parameters
    (weights, IDM, FDM) are estimated on the remaining (training) blocks and the
    strategy is evaluated out-of-sample on the held-out blocks, with an embargo
    gap purged around each test block to prevent train/test leakage.

    Unlike ``purged_walk_forward`` (a single causal path), CPCV yields a
    *distribution* of OOS Sharpe across C(n_groups, k_test) paths — a robustness
    measure that answers "how sensitive is the edge to which slice we hold out?"
    and enables a PBO-style tail read (fraction of paths with OOS Sharpe ≤ 0).

    NOTE: because a test block may sit *before* some training blocks, this is a
    robustness distribution, not a live-trading simulation — the purge/embargo
    bounds the adjacency leak but does not make every path strictly causal. Use
    it alongside, not instead of, ``purged_walk_forward``.
    """
    from .backtest import run_backtest, run_backtest_with_params
    from .metrics import sharpe

    prices = prices.sort_index().dropna(how="all")
    n = len(prices)
    if n_groups < 2 or k_test < 1 or k_test >= n_groups or n < 300:
        return {"insufficient": True, "n": n}

    bounds = np.linspace(0, n, n_groups + 1, dtype=int)
    groups = [(int(bounds[i]), int(bounds[i + 1])) for i in range(n_groups)]
    embargo = max(1, int(round(n * embargo_frac)))
    min_train = 128

    paths = []
    for test_ids in combinations(range(n_groups), k_test):
        test_mask = np.zeros(n, dtype=bool)
        purge_mask = np.zeros(n, dtype=bool)
        for gid in test_ids:
            lo, hi = groups[gid]
            test_mask[lo:hi] = True
            purge_mask[max(0, lo - embargo) : min(n, hi + embargo)] = True
        train_mask = ~purge_mask
        if train_mask.sum() < min_train or test_mask.sum() < 30:
            continue

        train = prices.iloc[train_mask]
        test = prices.iloc[test_mask]
        cot_train = cot.iloc[train_mask] if cot is not None else None
        cot_test = cot.iloc[test_mask] if cot is not None else None

        train_result = run_backtest(train, config, cot=cot_train)
        test_result = run_backtest_with_params(
            test,
            config,
            train_result.weights,
            train_result.idm,
            train_result.fdm,
            cot=cot_test,
        )
        paths.append(
            {
                "test_groups": list(test_ids),
                "is_sharpe": sharpe(train_result.daily_returns),
                "oos_sharpe": sharpe(test_result.daily_returns),
            }
        )

    if not paths:
        return {"insufficient": True, "n": n}

    oos = np.array([p["oos_sharpe"] for p in paths])
    gaps = np.array([p["is_sharpe"] - p["oos_sharpe"] for p in paths])
    return {
        "n_paths": len(paths),
        "n_groups": n_groups,
        "k_test": k_test,
        "paths": paths,
        "mean_oos_sharpe": float(oos.mean()),
        "median_oos_sharpe": float(np.median(oos)),
        "oos_p5": float(np.percentile(oos, 5)),
        "oos_p95": float(np.percentile(oos, 95)),
        "pct_paths_below_zero": float((oos <= 0).mean()),
        "mean_gap": float(gaps.mean()),
    }


def probability_backtest_overfitting(returns: pd.DataFrame, n_splits: int = 10) -> dict:
    """PBO via Combinatorially-Symmetric Cross-Validation (Bailey et al. 2017).

    ``returns`` is a T×N panel of daily returns, one column per *candidate
    configuration* from a strategy search (e.g. the rows of an experiment grid).
    PBO is the probability that the configuration selected as best in-sample lands
    below the median out-of-sample — i.e. that the search optimised noise.

    Algorithm: partition the rows into ``n_splits`` contiguous submatrices; for
    each way of assigning half to train and half to test, pick the IS-best config,
    find its OOS rank, map to a logit, and measure how often that logit is < 0.

    This is the direct, quantitative overfitting check the parent engine lacked:
    a high PBO (≳ 0.5) means the winning config is indistinguishable from luck.
    """
    r = returns.dropna(how="all").dropna(axis=1, how="any")
    t, n = r.shape
    if n < 2 or t < 2 * n_splits or n_splits < 2 or n_splits % 2 != 0:
        return {"insufficient": True, "n_configs": int(n), "n_obs": int(t)}

    bounds = np.linspace(0, t, n_splits + 1, dtype=int)
    blocks = [r.iloc[bounds[i] : bounds[i + 1]] for i in range(n_splits)]

    def _sharpe_row(m: pd.DataFrame) -> np.ndarray:
        mu = m.mean().to_numpy()
        sd = m.std(ddof=0).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(sd > 0, mu / sd, 0.0)

    logits = []
    for train_ids in combinations(range(n_splits), n_splits // 2):
        test_ids = [i for i in range(n_splits) if i not in train_ids]
        train = pd.concat([blocks[i] for i in train_ids])
        test = pd.concat([blocks[i] for i in test_ids])
        is_perf = _sharpe_row(train)
        oos_perf = _sharpe_row(test)
        best = int(np.argmax(is_perf))
        # Relative rank of the IS-best config among OOS performances, in (0,1).
        rank = float((oos_perf <= oos_perf[best]).sum()) / (n + 1)
        rank = min(max(rank, 1.0 / (n + 1)), n / (n + 1))
        logits.append(math.log(rank / (1.0 - rank)))

    logits = np.array(logits)
    return {
        "n_configs": int(n),
        "n_obs": int(t),
        "n_combinations": int(len(logits)),
        "pbo": float((logits < 0).mean()),
        "median_logit": float(np.median(logits)),
    }


# ── Honest trial counting ────────────────────────────────────────────────────
# The parent engine failed its own Deflated Sharpe once the *real* trial count was
# used.  The fix is to never hand-set n_trials: every config ever evaluated writes
# a fingerprint, and the deflation reads the registry length.


def config_fingerprint(config: Config) -> str:
    """Stable content hash of a Config (order-independent, tuple/dict-safe)."""
    payload = asdict(config) if is_dataclass(config) else dict(config)
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def register_trial(
    config: Config, label: str = "", path: Path | str = DEFAULT_TRIAL_REGISTRY
) -> str:
    """Record a config fingerprint in the trial registry (idempotent per config).

    Returns the fingerprint. Re-registering the same config is a no-op, so the
    registry length is the honest count of *distinct* strategies searched.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fp = config_fingerprint(config)
    if fp in _registry_fingerprints(path):
        return fp
    with open(path, "a") as f:
        f.write(json.dumps({"fingerprint": fp, "label": label, "desc": config.describe()}) + "\n")
    return fp


def _registry_fingerprints(path: Path) -> set[str]:
    if not Path(path).exists():
        return set()
    out = set()
    for line in Path(path).read_text().splitlines():
        if line.strip():
            out.add(json.loads(line)["fingerprint"])
    return out


def honest_n_trials(path: Path | str = DEFAULT_TRIAL_REGISTRY, floor: int = 1) -> int:
    """Number of distinct configs in the registry (the trial count to deflate by)."""
    return max(floor, len(_registry_fingerprints(Path(path))))


def assert_no_lookahead(
    fn,
    data: pd.DataFrame,
    truncate: int = 30,
    tol: float = 1e-9,
) -> dict:
    """Regression guard: an overlay/forecast must not change history when the
    future is revealed.

    ``fn(data) -> Series`` is evaluated on the full panel and again on the panel
    with the last ``truncate`` rows removed. On the shared (earlier) dates the two
    outputs must agree to ``tol`` — if they don't, ``fn`` is peeking ahead (e.g. a
    smoothed HMM posterior, a full-sample scalar, a centred rolling window).

    Returns a report; raises AssertionError on a leak. Use in tests to lock the
    regime overlays and any new signal as causal.
    """
    full = pd.Series(fn(data)).dropna()
    truncated = pd.Series(fn(data.iloc[:-truncate])).dropna()
    shared = full.index.intersection(truncated.index)
    if len(shared) == 0:
        return {"insufficient": True, "n_shared": 0}
    max_diff = float((full.loc[shared] - truncated.loc[shared]).abs().max())
    leaked = max_diff > tol
    if leaked:
        raise AssertionError(
            f"lookahead detected: history changed by {max_diff:.2e} (> {tol:.0e}) "
            f"when {truncate} future rows were revealed"
        )
    return {"n_shared": int(len(shared)), "max_diff": max_diff, "causal": True}
