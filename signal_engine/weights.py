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

from .markets import BY_SYMBOL


def _asset_class(sym: str) -> str:
    inst = BY_SYMBOL.get(sym)
    return inst.asset_class if inst else "other"


def cluster_weights(
    symbols: list[str], class_of: Callable[[str], str] | None = None
) -> dict[str, float]:
    """Risk equally across asset-class clusters, then equally within each.

    Weights sum to 1.0; every cluster gets 1/n_clusters regardless of size.
    """
    class_of = class_of or _asset_class
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
