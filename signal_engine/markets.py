"""The instrument universe — ~20 markets across 5 asset classes.

Diversification is the whole edge, so the universe is chosen for LOW mutual
correlation across asset classes, not for individual attractiveness.

Data honesty (the §84 lesson, applied from day one)
---------------------------------------------------
Proper continuous futures with back-adjusted prices AND term structure (needed
for real carry) require a paid feed (CSI / Norgate / a broker). The free default
here uses liquid ETF proxies, which capture the same trend betas cleanly on free
yfinance data. Carry on the ETF path is therefore limited — see README §carry.
Swap `kind="future"` + real symbols when a paid feed is available; the engine is
agnostic to which it gets.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    symbol: str  # data symbol (ETF proxy by default)
    name: str
    asset_class: str  # equity | bond | commodity | fx | credit | real_estate
    multiplier: float = 1.0  # contract multiplier; 1.0 for ETF shares
    kind: str = "etf"  # "etf" | "future"
    carry_kind: str | None = None  # "bond_slope" | "term_structure" | None
    cost_bps: float = 1.5  # per-side cost in bps of notional traded


# Curated low-correlation set. ETF proxies → clean free data.
# Per-instrument costs are rough per-side spreads for the ETF proxies; the
# default run still uses the flat Config.cost_bps unless --cost-scheme instrument.
UNIVERSE: tuple[Instrument, ...] = (
    # Equity (geographically diversified so they aren't one bet)
    Instrument("SPY", "US large cap", "equity", cost_bps=1.0),
    Instrument("IWM", "US small cap", "equity", cost_bps=1.5),
    Instrument("EFA", "Developed ex-US", "equity", cost_bps=2.0),
    Instrument("EEM", "Emerging markets", "equity", cost_bps=3.0),
    Instrument("EWJ", "Japan", "equity", cost_bps=2.0),
    # Rates / bonds
    Instrument("TLT", "US 20y+ Treasury", "bond", carry_kind="bond_slope", cost_bps=1.0),
    Instrument("IEF", "US 7-10y Treasury", "bond", carry_kind="bond_slope", cost_bps=1.0),
    Instrument("TIP", "US TIPS", "bond", carry_kind="bond_slope", cost_bps=1.0),
    # Commodities
    Instrument("GLD", "Gold", "commodity", cost_bps=1.0),
    Instrument("SLV", "Silver", "commodity", cost_bps=2.5),
    Instrument("DBC", "Broad commodities", "commodity", cost_bps=2.0),
    Instrument("USO", "WTI crude", "commodity", cost_bps=3.0),
    Instrument("DBA", "Agriculture", "commodity", cost_bps=2.5),
    # FX (vs USD)
    Instrument("UUP", "US dollar index", "fx"),
    Instrument("FXE", "Euro", "fx"),
    Instrument("FXY", "Japanese yen", "fx"),
    # Credit
    Instrument("HYG", "US high yield", "credit", cost_bps=2.0),
    Instrument("LQD", "US investment grade", "credit", cost_bps=1.5),
    # Real estate
    Instrument("VNQ", "US REITs", "real_estate", cost_bps=2.5),
)

# Opt-in instrument packs that can be added to the core universe via CLI flags.
# They are NOT in UNIVERSE (the deployed 19), but they carry metadata so that
# reports and per-instrument costs work when a user adds them.
OPTION_PACK_INSTRUMENTS: tuple[Instrument, ...] = (
    Instrument("QQQ", "Nasdaq-100", "equity", cost_bps=1.0),
    Instrument("SMH", "Semiconductors", "equity", cost_bps=2.5),
    Instrument("SOXX", "Semiconductors", "equity", cost_bps=2.5),
    Instrument("XSD", "Semiconductors", "equity", cost_bps=2.5),
    Instrument("BNDX", "Intl aggregate bonds", "bond", cost_bps=2.0),
    Instrument("MUB", "Municipal bonds", "bond", cost_bps=1.5),
    Instrument("EMLC", "EM local currency bonds", "bond", cost_bps=2.5),
    Instrument("PFF", "Preferred stocks", "credit", cost_bps=2.0),
    Instrument("AMLP", "MLPs / energy infrastructure", "commodity", cost_bps=2.5),
)

# Expanded free-ETF universe. Thin/young ETFs are automatically filtered by the
# >300-bar rule in data.load_prices, so adding candidates is cheap.
EXPANDED_UNIVERSE: tuple[Instrument, ...] = UNIVERSE + OPTION_PACK_INSTRUMENTS + (
    # More equity regions / factors
    Instrument("FXI", "China large cap", "equity", cost_bps=2.5),
    Instrument("EWG", "Germany", "equity", cost_bps=2.0),
    Instrument("EWZ", "Brazil", "equity", cost_bps=3.0),
    Instrument("INDA", "India", "equity", cost_bps=3.0),
    Instrument("EWC", "Canada", "equity", cost_bps=2.0),
    Instrument("EWT", "Taiwan", "equity", cost_bps=2.5),
    Instrument("EWY", "South Korea", "equity", cost_bps=2.5),
    Instrument("EIDO", "Indonesia", "equity", cost_bps=3.5),
    # More rates / credit
    Instrument("SHY", "US 1-3y Treasury", "bond", carry_kind="bond_slope", cost_bps=1.0),
    Instrument("EMB", "USD emerging-market bonds", "bond", cost_bps=2.5),
    Instrument("BWX", "Intl Treasury", "bond", cost_bps=2.0),
    Instrument("JNK", "US high yield", "credit", cost_bps=2.0),
    Instrument("VCIT", "Intermediate corp credit", "credit", cost_bps=1.5),
    # More commodities
    Instrument("CORN", "Corn", "commodity", cost_bps=3.0),
    Instrument("WEAT", "Wheat", "commodity", cost_bps=3.0),
    Instrument("SOYB", "Soybeans", "commodity", cost_bps=3.0),
    Instrument("UNG", "Natural gas", "commodity", cost_bps=3.0),
    Instrument("CPER", "Copper", "commodity", cost_bps=2.5),
    # More FX
    Instrument("FXB", "British pound", "fx"),
    Instrument("FXA", "Australian dollar", "fx"),
    Instrument("FXC", "Canadian dollar", "fx"),
    Instrument("FXS", "Swedish krona", "fx"),
    # Global real estate
    Instrument("REET", "Global REITs", "real_estate", cost_bps=2.5),
)

# Synthetic instruments that are injected from free data (e.g., FRED curve trades).
# They are NOT in the price-fetch symbol lists, but they carry metadata so the
# engine knows how to cost and classify them once merged into the price panel.
SYNTHETIC_INSTRUMENTS: tuple[Instrument, ...] = (
    Instrument(
        "UST2S10S",
        "US 2s10s curve steepener",
        "bond",
        carry_kind="curve_steepener",
        cost_bps=0.5,
    ),
    Instrument(
        "SP500_XSMOM",
        "S&P 500 cross-sectional momentum sleeve",
        "equity",
        carry_kind=None,
        cost_bps=5.0,
    ),
)

BY_SYMBOL: dict[str, Instrument] = {
    i.symbol: i for i in UNIVERSE + SYNTHETIC_INSTRUMENTS + OPTION_PACK_INSTRUMENTS
}
BY_SYMBOL_EXPANDED: dict[str, Instrument] = {
    i.symbol: i for i in EXPANDED_UNIVERSE + SYNTHETIC_INSTRUMENTS
}


def symbols(expanded: bool = False) -> list[str]:
    src = EXPANDED_UNIVERSE if expanded else UNIVERSE
    return [i.symbol for i in src]


def asset_classes(expanded: bool = False) -> dict[str, list[str]]:
    """symbol → group, used for cluster-aware weighting and reporting."""
    src = EXPANDED_UNIVERSE if expanded else UNIVERSE
    out: dict[str, list[str]] = {}
    for inst in src:
        out.setdefault(inst.asset_class, []).append(inst.symbol)
    return out


def instrument_for(symbol: str, expanded: bool = False) -> Instrument | None:
    """Return the Instrument metadata for a symbol, or None if unknown."""
    src = BY_SYMBOL_EXPANDED if expanded else BY_SYMBOL
    return src.get(symbol)


def cost_per_symbol(expanded: bool = False) -> dict[str, float]:
    """symbol → per-side cost in bps."""
    src = BY_SYMBOL_EXPANDED if expanded else BY_SYMBOL
    return {sym: inst.cost_bps for sym, inst in src.items()}
