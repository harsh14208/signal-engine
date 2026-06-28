"""Instrument risk weights.

Equal-weighting this universe is a trap: 5 of the ~19 instruments are equity
ETFs that move together, so naive 1/N hands the *single* "long equities" bet
~5× the risk budget of the lone REIT. Two-level "handcrafting" (Carver) fixes
it: split risk equally across asset-class clusters, then equally within each
cluster — so the whole equity sleeve counts as ONE bet.

(A correlation-clustered version is the natural next refinement; asset-class
clusters are the interpretable v1 and match how the universe was chosen.)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .config import Config
from .markets import instrument_for


def _asset_class(sym: str, expanded: bool = False) -> str:
    inst = instrument_for(sym, expanded)
    return inst.asset_class if inst else "other"


def cluster_weights(
    symbols: list[str],
    class_of: Callable[[str], str] | None = None,
    expanded: bool = False,
) -> dict[str, float]:
    """Risk equally across asset-class clusters, then equally within each.

    Weights sum to 1.0; every cluster gets 1/n_clusters regardless of size.
    """
    class_of = class_of or (lambda s: _asset_class(s, expanded))
    clusters: dict[str, list[str]] = {}
    for s in symbols:
        clusters.setdefault(class_of(s), []).append(s)
    if not clusters:
        return {}
    per_cluster = 1.0 / len(clusters)
    out: dict[str, float] = {}
    for members in clusters.values():
        for m in members:
            out[m] = per_cluster / len(members)
    return out


class _UnionFind:
    """Tiny union-find for correlation-threshold clustering."""

    def __init__(self, items: list[str]):
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        # Path compression.
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def clusters(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for x in self.parent:
            out.setdefault(self.find(x), []).append(x)
        return out


def corr_cluster_weights(
    symbols: list[str], returns: pd.DataFrame, threshold: float = 0.5
) -> dict[str, float]:
    """Carver-style handcrafting by correlation clusters.

    Instruments with pairwise correlation above `threshold` are grouped;
    risk is split equally across clusters, then equally within each cluster.
    Unlike asset-class clustering, this cannot create a singleton cluster
    that inflates a weak name's budget.
    """
    symbols = [s for s in symbols if s in returns.columns]
    if not symbols:
        return {}
    if len(symbols) == 1:
        return {symbols[0]: 1.0}

    corr_arr = returns[symbols].corr().fillna(0.0).to_numpy().copy()
    np.fill_diagonal(corr_arr, 1.0)
    uf = _UnionFind(symbols)
    for i, a in enumerate(symbols):
        for j, b in enumerate(symbols[i + 1 :], start=i + 1):
            if corr_arr[i, j] >= threshold:
                uf.union(a, b)
    clusters = uf.clusters()
    per_cluster = 1.0 / len(clusters)
    out: dict[str, float] = {}
    for members in clusters.values():
        for m in members:
            out[m] = per_cluster / len(members)
    return out


def _expanding_sharpe(returns: pd.Series) -> float:
    """Full-sample annualised Sharpe used for static weight calibration."""
    r = returns.dropna()
    if len(r) < 30 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(256))


def sharpe_adjusted_weights(
    symbols: list[str],
    returns: pd.DataFrame,
    base_weights: dict[str, float] | None = None,
    floor: float = 0.0,
) -> dict[str, float]:
    """Down-weight chronically negative-Sharpe instruments.

    Starting from `base_weights` (equal if not provided), each weight is
    multiplied by max(floor, Sharpe) and the result is renormalised to 1.0.
    """
    symbols = [s for s in symbols if s in returns.columns]
    if not symbols:
        return {}
    if base_weights is None:
        base_weights = {s: 1.0 / len(symbols) for s in symbols}
    scores = {s: max(floor, _expanding_sharpe(returns[s])) for s in symbols}
    raw = {s: base_weights.get(s, 0.0) * scores[s] for s in symbols}
    total = sum(raw.values())
    if total <= 0:
        return {s: 1.0 / len(symbols) for s in symbols}
    return {s: raw[s] / total for s in symbols}


def equal_weights(symbols: list[str]) -> dict[str, float]:
    """Naive 1/N weights."""
    if not symbols:
        return {}
    w = 1.0 / len(symbols)
    return {s: w for s in symbols}


def build_instrument_weights(
    symbols: list[str],
    returns: pd.DataFrame,
    config: Config,
    expanded: bool | None = None,
) -> dict[str, float]:
    """Dispatch to the weighting scheme selected in `config`."""
    if expanded is None:
        expanded = getattr(config, "use_expanded_universe", False)
    scheme = config.weight_scheme
    if config.cluster_weights and scheme == "equal":
        scheme = "cluster"

    if scheme == "cluster":
        return cluster_weights(symbols, expanded=expanded)
    if scheme == "corr_cluster":
        return corr_cluster_weights(symbols, returns)
    if scheme == "sharpe":
        return sharpe_adjusted_weights(symbols, returns)
    return equal_weights(symbols)
