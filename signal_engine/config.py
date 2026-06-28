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

# Optional GARCH(1,1) forward-vol estimate (requires `pip install -e .[garch]`)
VOL_GARCH_MIN_HISTORY = 252
VOL_GARCH_REFIT_STEP = 63
VOL_GARCH_HORIZON = 1
VOL_GARCH_DIST = "t"

# ── Risk / sizing ────────────────────────────────────────────────────────────
DEFAULT_CAPITAL = 1_000_000.0
DEFAULT_VOL_TARGET = 0.20  # annualised portfolio vol target (20%)

# ── Costs / turnover ─────────────────────────────────────────────────────────
DEFAULT_COST_BPS = 1.5  # per-side cost in bps of notional traded
BUFFER_FRACTION = 0.30  # position buffer (× avg position) to cut turnover

# ── Realised-vol governor ────────────────────────────────────────────────────
# Forecast scalars are calibrated for futures, and the IDM assumes correlations
# stay put — both make realised vol drift above target on real data. The governor
# is a feedback overlay that scales the whole book by target/trailing-realised-vol
# so realised vol actually lands near the target (and the tail shrinks with it).
GOVERNOR_SPAN = 32  # trailing EW span for the realised-vol estimate
GOVERNOR_MIN = 0.20  # clamp the leverage multiplier (avoid blow-up at low vol)
GOVERNOR_MAX = 2.50


@dataclass
class Config:
    """Runtime knobs. Defaults reproduce the canonical diversified run."""

    capital: float = DEFAULT_CAPITAL
    vol_target: float = DEFAULT_VOL_TARGET
    ewmac_speeds: tuple[tuple[int, int], ...] = DEFAULT_EWMAC_SPEEDS
    breakout_spans: tuple[int, ...] = DEFAULT_BREAKOUT_SPANS
    use_breakout: bool = True
    use_carry: bool = False  # legacy synthetic/demo carry flag
    use_carry_proxies: bool = False  # free bond/equity carry proxies
    use_real_bond_carry: bool = False  # tenor-specific roll-down carry from DGS curve
    use_curve_steepener: bool = False  # add UST 2s10s synthetic instrument
    curve_steepener_scale: float = 1.0
    use_equity_momentum_sleeve: bool = False  # add SP500 cross-sectional momentum sleeve
    eq_mom_lookback: int = 252
    eq_mom_rebalance: int = 21
    eq_mom_decile: float = 0.10
    use_expanded_universe: bool = False
    use_empirical_scalars: bool = False
    use_regime_overlay: bool = False
    regime_threshold: float = 20.0
    regime_max_degear: float = 0.5
    use_hmm_regime_overlay: bool = False
    hmm_train_window: int = 252
    hmm_refit_stride: int = 63
    hmm_bull_thresh: float = 0.75
    hmm_bear_thresh: float = 0.70
    hmm_trans_thresh: float = 0.15
    hmm_bull_gear: float = 1.10
    hmm_bear_degear: float = 0.70
    hmm_trans_degear: float = 0.85
    hmm_smooth: int | None = None
    use_vix_term_overlay: bool = False
    vix_term_short_thresh: float = 1.10
    vix_term_long_thresh: float = 0.95
    vix_term_max_gear: float = 1.25
    vix_term_max_degear: float = 0.50
    use_credit_overlay: bool = False
    credit_upper_thresh: float = 1.50
    credit_lower_thresh: float = 0.80
    credit_lookback: int = 1260
    credit_max_gear: float = 1.25
    credit_max_degear: float = 0.50
    cost_bps: float = DEFAULT_COST_BPS
    cost_scheme: str = "flat"  # "flat" uses cost_bps; "instrument" uses per-Instrument costs
    buffer_fraction: float = BUFFER_FRACTION
    fdm_cap: float = FDM_CAP
    idm_cap: float = IDM_CAP
    forecast_cap: float = FORECAST_CAP
    # Weighting.  "equal" | "cluster" | "corr_cluster" | "sharpe".
    # cluster_weights is retained for backward compatibility and maps to "cluster".
    weight_scheme: str = "equal"
    cluster_weights: bool = False
    use_governor: bool = True  # realised-vol overlay; ablation win (Calmar 0.23→0.34)
    governor_span: int = GOVERNOR_SPAN
    governor_min: float = GOVERNOR_MIN
    governor_max: float = GOVERNOR_MAX
    governor_smooth: int | None = None  # EWMA span on the leverage multiplier; None = raw
    regime_smooth: int | None = None  # EWMA span on the regime de-gross multiplier; None = raw
    # Optional GARCH(1,1) forward-vol estimate for the vol denominator.
    use_garch_vol: bool = False
    garch_weight: float = 0.0  # 0 = pure EWMA blend, 1 = pure GARCH
    garch_min_history: int = VOL_GARCH_MIN_HISTORY
    garch_refit_step: int = VOL_GARCH_REFIT_STEP
    garch_horizon: int = VOL_GARCH_HORIZON
    hmm_random_state: int = 42
    curve_steepener_cost_bps: float = 0.5
    vix_term_smooth: int | None = None  # EWMA span on the VIX term-structure multiplier; None = raw
    credit_smooth: int | None = None  # EWMA span on the credit multiplier; None = raw
    rule_weights: dict[str, float] = field(default_factory=dict)  # empty → equal
    # Additional orthogonal rules (off by default; validate each vs placebo before shipping).
    use_accel: bool = False
    accel_speeds: tuple[tuple[int, int], ...] = ((8, 32), (16, 64))
    use_xsmom: bool = False
    xsmom_lookback: int = 64
    # Correlation-spike de-risking overlay.
    use_corr_spike: bool = False
    corr_spike_span: int = 60
    corr_spike_threshold: float = 0.50
    corr_spike_max_degross: float = 0.50
    # COT (CFTC Commitments of Traders) positioning forecast — free, weekly, 1986+.
    use_cot: bool = False
    cot_momentum: bool = False  # flip COT sign contrarian→momentum (research A/B only)
    # Surgical "core + diversifying-commodity COT" universe (adds UNG/CORN/WEAT to core).
    use_core_commodities: bool = False
    # Out-of-sample parameter calibration.  Parameters (instrument weights, IDM,
    # FDM) are re-estimated on an expanding window ending at each rebal point,
    # then applied forward only.  This removes the full-sample calibration leak in
    # the default run_backtest path.
    calibration_min_obs: int = 256
    calibration_rebal: int = 252

    def __post_init__(self):
        # Backward compatibility: the old boolean flag overrides the scheme.
        if self.cluster_weights and self.weight_scheme == "equal":
            object.__setattr__(self, "weight_scheme", "cluster")

    def describe(self) -> str:
        rules = [f"EWMAC{s}" for s in self.ewmac_speeds]
        if self.use_breakout:
            rules += [f"BO{n}" for n in self.breakout_spans]
        if self.use_carry or self.use_carry_proxies:
            rules += ["CARRY"]
        if self.use_accel:
            rules += ["ACCEL"]
        if self.use_xsmom:
            rules += ["XSMOM"]
        if self.use_cot:
            rules += ["COT"]
        weight = self.weight_scheme
        if self.weight_scheme == "cluster" or self.cluster_weights:
            weight = "cluster"
        smooth = f" smooth={self.governor_smooth}" if self.governor_smooth else ""
        regime_smooth = f" regime_smooth={self.regime_smooth}" if self.regime_smooth else ""
        vix_term = "on" if self.use_vix_term_overlay else "off"
        vix_term_smooth = f" vts_smooth={self.vix_term_smooth}" if self.vix_term_smooth else ""
        credit = "on" if self.use_credit_overlay else "off"
        credit_smooth = f" credit_smooth={self.credit_smooth}" if self.credit_smooth else ""
        garch = f"garch_w={self.garch_weight:.0%}" if self.use_garch_vol else "ewma"
        hmm = "on" if self.use_hmm_regime_overlay else "off"
        bond_carry = (
            "real" if self.use_real_bond_carry else ("proxies" if self.use_carry_proxies else "off")
        )
        curve = "on" if self.use_curve_steepener else "off"
        eq_mom = "on" if self.use_equity_momentum_sleeve else "off"
        carry = "proxies" if self.use_carry_proxies else ("on" if self.use_carry else "off")
        return (
            f"capital=${self.capital:,.0f} vol_target={self.vol_target:.0%} "
            f"cost={self.cost_bps}bps scheme={self.cost_scheme} "
            f"buffer={self.buffer_fraction:.0%} weights={weight} "
            f"universe={'expanded' if self.use_expanded_universe else 'core'} "
            f"governor={'on' if self.use_governor else 'off'}{smooth} "
            f"regime={'on' if self.use_regime_overlay else 'off'}{regime_smooth} "
            f"vix_term={vix_term}{vix_term_smooth} "
            f"credit={credit}{credit_smooth} "
            f"hmm={hmm} "
            f"bond_carry={bond_carry} curve={curve} eq_mom={eq_mom} "
            f"carry={carry} scalars={'empirical' if self.use_empirical_scalars else 'fixed'} "
            f"corr_spike={'on' if self.use_corr_spike else 'off'} "
            f"vol={garch} "
            f"rules=[{', '.join(rules)}]"
        )
