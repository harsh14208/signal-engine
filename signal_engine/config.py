"""Engine constants and the runtime `Config` object.

All magic numbers live here, not scattered through the code — the antithesis of
a 5,800-line engine with constants buried in 96 gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ── Annualisation ────────────────────────────────────────────────────────────
BUSINESS_DAYS_YEAR = 256  # Carver convention
ANNUAL_VOL_SQRT = 16.0  # sqrt(256)

# ── Forecast scaling (Carver) ────────────────────────────────────────────────
AVG_ABS_FORECAST = 10.0  # every rule scaled so mean(|forecast|) ≈ 10
FORECAST_CAP = 20.0  # clip at ±2× the average

# Published EWMAC forecast scalars (estimated once across many instruments and
# treated as CONSTANTS — using an in-sample empirical scalar would be lookahead).
EWMAC_SCALARS: dict[tuple[int, int], float] = {
    (8, 32): 5.3,
    (16, 64): 3.75,
    (32, 128): 2.65,
    (64, 256): 1.87,
}
DEFAULT_EWMAC_SPEEDS: tuple[tuple[int, int], ...] = ((16, 64), (32, 128), (64, 256))

# Breakout (secondary trend rule). Scalars approximate Carver's published set.
BREAKOUT_SCALARS: dict[int, float] = {20: 30.0, 40: 31.0, 80: 33.0, 160: 33.0}
DEFAULT_BREAKOUT_SPANS: tuple[int, ...] = (40, 80, 160)

# Carry forecast scalar (Carver ≈ 30 for risk-adjusted annualised carry).
CARRY_SCALAR = 30.0

# Diversification multipliers — combining correlated signals/instruments shrinks
# variance, so the combined forecast/position is scaled back up, capped for safety.
FDM_CAP = 2.5  # forecast diversification multiplier cap
IDM_CAP = 2.5  # instrument diversification multiplier cap

# ── Volatility estimation ────────────────────────────────────────────────────
VOL_EW_SPAN = 32  # recent exponentially-weighted daily-return vol
VOL_LONG_WEIGHT = 0.30  # blend: 30% long-run avg + 70% recent (Carver)
VOL_MIN_PERIODS = 20

# ── Risk / sizing ────────────────────────────────────────────────────────────
DEFAULT_CAPITAL = 1_000_000.0
DEFAULT_VOL_TARGET = 0.20  # annualised portfolio vol target (20%)

# ── Costs / turnover ─────────────────────────────────────────────────────────
DEFAULT_COST_BPS = 1.5  # per-side cost in bps of notional traded
BUFFER_FRACTION = 0.10  # position buffer (× avg position) to cut turnover


@dataclass
class Config:
    """Runtime knobs. Defaults reproduce the canonical diversified run."""

    capital: float = DEFAULT_CAPITAL
    vol_target: float = DEFAULT_VOL_TARGET
    ewmac_speeds: tuple[tuple[int, int], ...] = DEFAULT_EWMAC_SPEEDS
    breakout_spans: tuple[int, ...] = DEFAULT_BREAKOUT_SPANS
    use_breakout: bool = True
    use_carry: bool = False  # needs term-structure data (see README §carry)
    cost_bps: float = DEFAULT_COST_BPS
    buffer_fraction: float = BUFFER_FRACTION
    fdm_cap: float = FDM_CAP
    idm_cap: float = IDM_CAP
    forecast_cap: float = FORECAST_CAP
    rule_weights: dict[str, float] = field(default_factory=dict)  # empty → equal

    def describe(self) -> str:
        rules = [f"EWMAC{s}" for s in self.ewmac_speeds]
        if self.use_breakout:
            rules += [f"BO{n}" for n in self.breakout_spans]
        if self.use_carry:
            rules += ["CARRY"]
        return (
            f"capital=${self.capital:,.0f} vol_target={self.vol_target:.0%} "
            f"cost={self.cost_bps}bps buffer={self.buffer_fraction:.0%} "
            f"rules=[{', '.join(rules)}]"
        )
