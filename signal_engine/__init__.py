"""signal-engine — diversified systematic futures trend + carry.

The one-sentence edge thesis
----------------------------
Many small, *uncorrelated* risk-adjusted bets (trend + carry across asset
classes) stack into a portfolio Sharpe far higher than any single bet — because
diversification is the only free lunch. This is the opposite of a single
over-fit equity strategy: no instrument here is expected to be impressive on
its own (~0.2–0.3 standalone Sharpe); the edge lives in combining ~20 weakly
correlated return streams at a constant risk target.

Build discipline (the sequencing fix)
-------------------------------------
This package is a *research harness first*. It proves the edge out-of-sample
with the rigor tooling in `validation.py` BEFORE any live/execution layer is
written. Complexity is added only when a pre-registered holdout says it pays.
"""

from .config import Config

__all__ = ["Config"]
__version__ = "0.1.0"
