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
    # "flat" uses cost_bps; "instrument" uses per-Instrument costs; "calibrated"
    # reads measured per-side costs from the friction calibration file.
    cost_scheme: str = "flat"
    calibration_path: str | None = None  # override friction.DEFAULT_CALIBRATION_PATH
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
    # Network ("follow the leader") momentum — price-only lead-lag graph signal.
    use_network_momentum: bool = False
    nm_speed: tuple[int, int] = (32, 128)
    nm_lookback: int = 256
    nm_lag: int = 1
    nm_rebal: int = 63
    nm_top_k: int = 3
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
    # Number of days over which to smooth calibration transitions (weights/IDM/FDM).
    # None = instantaneous jump at each rebal date. A small window (e.g. 10-20)
    # cuts estimation-driven turnover without adding lookahead.
    calibration_smooth: int | None = None
    # Gross exposure cap. None = uncapped. Applied after the governor; a value of
    # 1.0 means no leverage, 3.0 means 3× capital in absolute notional.
    max_gross_notional: float | None = None
    # Financing cost on levered gross exposure. Charged on the portion of gross
    # notional above `financing_threshold` × capital, at `financing_rate` annual.
    # 0.0 = no financing cost. A realistic retail margin spread is ~1.0–1.5%.
    financing_rate: float = 0.0
    financing_threshold: float = 1.0
    # Optional hard cap on annual financing cost as a fraction of capital.
    # If set, positions are scaled down so expected annual financing
    # (gross above threshold * financing_rate) does not exceed this amount.
    max_annual_financing_cost: float | None = None

    # Drawdown-state control (opt-in). Scales positions down when the strategy
    # hits a realised drawdown threshold, and scales back up on recovery.
    use_drawdown_control: bool = False
    drawdown_threshold: float = 0.10  # e.g. 10% drawdown triggers de-risk
    drawdown_scale: float = 0.50      # scale positions to 50% when triggered
    drawdown_recovery: float = 0.05   # re-risk when drawdown recovers to 5%

    # Trend-strength filter (opt-in). De-gears the book when the average absolute
    # combined forecast falls into the bottom percentile of its recent history — a
    # response to the 2023-26 weak-trend environment. Must be validated OOS to avoid
    # overfitting the recent past.
    use_trend_strength_filter: bool = False
    trend_strength_window: int = 63
    trend_strength_threshold: float = 0.25  # bottom quartile
    trend_strength_scale: float = 0.70      # scale positions to 70% when weak

    # Crypto pack (opt-in). BTC + ETH via yfinance.  High vol, short history,
    # low correlation to traditional assets.  `crypto_max_risk_weight` caps the
    # portfolio-level risk contribution of any single crypto instrument so it
    # cannot dominate the IDM.
    use_crypto: bool = False
    crypto_max_risk_weight: float = 0.05  # max risk contribution per crypto symbol

    # Curated breadth pack (opt-in). Correlation-selected additions to the core
    # universe (commodities, EM/intl bonds, FX) — a salvage of the failed
    # --expanded-universe test, which added too many correlated equity betas.
    use_curated_breadth: bool = False

    # FX carry from FRED rate differentials (opt-in). Replaces the dividend-yield
    # proxy for FX ETFs with (foreign_short_rate - usd_short_rate), the classic
    # free FX carry signal.
    use_fx_carry: bool = False

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
        if self.use_network_momentum:
            rules += ["NETMOM"]
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
        parts = [
            f"capital=${self.capital:,.0f} vol_target={self.vol_target:.0%}",
            f"cost={self.cost_bps}bps scheme={self.cost_scheme}",
            f"buffer={self.buffer_fraction:.0%} weights={weight}",
            f"universe={'expanded' if self.use_expanded_universe else 'core'}",
            f"crypto={'on' if self.use_crypto else 'off'}",
            f"curated_breadth={'on' if self.use_curated_breadth else 'off'}",
            f"fx_carry={'on' if self.use_fx_carry else 'off'}",
            f"governor={'on' if self.use_governor else 'off'}{smooth}",
            f"regime={'on' if self.use_regime_overlay else 'off'}{regime_smooth}",
            f"vix_term={vix_term}{vix_term_smooth}",
            f"credit={credit}{credit_smooth}",
            f"hmm={hmm}",
            f"bond_carry={bond_carry} curve={curve} eq_mom={eq_mom}",
            f"carry={carry} scalars={'empirical' if self.use_empirical_scalars else 'fixed'}",
            f"corr_spike={'on' if self.use_corr_spike else 'off'}",
            f"vol={garch}",
            f"rules=[{', '.join(rules)}]",
        ]
        if self.max_gross_notional is not None:
            parts.append(f"max_gross={self.max_gross_notional:.1f}x")
        if self.financing_rate != 0.0:
            parts.append(f"fin={self.financing_rate:.2%}")
        return " ".join(parts)

    def min_history_required(self) -> int:
        """Minimum price observations needed for a warm, deterministic restart.

        This is the longest lookback any enabled rule or overlay requires. A
        restarted loop with fewer observations will produce different indicator
        state than a continuously-running loop.
        """
        lookbacks = [self.calibration_min_obs, GOVERNOR_SPAN, VOL_MIN_PERIODS]
        if self.use_breakout:
            lookbacks.extend(self.breakout_spans)
        for fast, slow in self.ewmac_speeds:
            lookbacks.append(slow)
        if self.use_corr_spike:
            lookbacks.append(self.corr_spike_span)
        if self.use_hmm_regime_overlay:
            lookbacks.append(self.hmm_train_window)
        if self.use_garch_vol:
            lookbacks.append(self.garch_min_history)
        if self.use_regime_overlay:
            lookbacks.append(self.regime_smooth or 1)
        if self.eq_mom_lookback:
            lookbacks.append(self.eq_mom_lookback)
        return max(lookbacks)
