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


# Curated low-correlation set. ETF proxies → clean free data.
UNIVERSE: tuple[Instrument, ...] = (
    # Equity (geographically diversified so they aren't one bet)
    Instrument("SPY", "US large cap", "equity"),
    Instrument("IWM", "US small cap", "equity"),
    Instrument("EFA", "Developed ex-US", "equity"),
    Instrument("EEM", "Emerging markets", "equity"),
    Instrument("EWJ", "Japan", "equity"),
    # Rates / bonds
    Instrument("TLT", "US 20y+ Treasury", "bond", carry_kind="bond_slope"),
    Instrument("IEF", "US 7-10y Treasury", "bond", carry_kind="bond_slope"),
    Instrument("TIP", "US TIPS", "bond", carry_kind="bond_slope"),
    # Commodities
    Instrument("GLD", "Gold", "commodity"),
    Instrument("SLV", "Silver", "commodity"),
    Instrument("DBC", "Broad commodities", "commodity"),
    Instrument("USO", "WTI crude", "commodity"),
    Instrument("DBA", "Agriculture", "commodity"),
    # FX (vs USD)
    Instrument("UUP", "US dollar index", "fx"),
    Instrument("FXE", "Euro", "fx"),
    Instrument("FXY", "Japanese yen", "fx"),
    # Credit
    Instrument("HYG", "US high yield", "credit"),
    Instrument("LQD", "US investment grade", "credit"),
    # Real estate
    Instrument("VNQ", "US REITs", "real_estate"),
)

BY_SYMBOL: dict[str, Instrument] = {i.symbol: i for i in UNIVERSE}


def symbols() -> list[str]:
    return [i.symbol for i in UNIVERSE]


def asset_classes() -> dict[str, list[str]]:
    """symbol → group, used for cluster-aware weighting and reporting."""
    out: dict[str, list[str]] = {}
    for inst in UNIVERSE:
        out.setdefault(inst.asset_class, []).append(inst.symbol)
    return out
