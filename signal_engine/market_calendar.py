"""Phase 2 — NYSE trading calendar (dependency-free).

The forward loop currently just "records nothing" on a holiday and can't tell a
market closure from a *data gap* (a session that should exist but is missing from
the panel — the exact silent-data-breakage class the parent engine died on). This
module gives the expected NYSE sessions so gaps become detectable.

Built on pandas' holiday primitives, so no extra dependency. Uses `nearest_workday`
observance — the standard NYSE approximation; good for gap detection, not intended
as a settlement-grade calendar.
"""

from __future__ import annotations

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)


class NYSECalendar(AbstractHolidayCalendar):
    """Full-day NYSE market closures (excludes early-close half days)."""

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        # NYSE first observed Juneteenth on 2022-06-20 (not 2021).
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


# Ad-hoc, non-recurring full-day closures a rules calendar can't derive
# (disasters, national days of mourning). Extend as new ones occur.
AD_HOC_CLOSURES: tuple[str, ...] = (
    "2001-09-11", "2001-09-12", "2001-09-13", "2001-09-14",  # 9/11
    "2004-06-11",  # Reagan day of mourning
    "2007-01-02",  # Ford day of mourning
    "2012-10-29", "2012-10-30",  # Hurricane Sandy
    "2018-12-05",  # George H.W. Bush day of mourning
    "2025-01-09",  # Jimmy Carter day of mourning
)

_CAL = NYSECalendar()
_AD_HOC = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in AD_HOC_CLOSURES))


def holidays(start, end) -> pd.DatetimeIndex:
    """NYSE full-day closures in [start, end] (recurring rules + ad-hoc closures)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    rule_hols = _CAL.holidays(start, end)
    ad_hoc = _AD_HOC[(_AD_HOC >= start) & (_AD_HOC <= end)]
    return rule_hols.union(ad_hoc)


def trading_days(start, end) -> pd.DatetimeIndex:
    """Expected NYSE trading sessions in [start, end] (weekdays minus holidays)."""
    days = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))
    return days.difference(holidays(start, end))


def is_trading_day(date) -> bool:
    d = pd.Timestamp(date).normalize()
    return d in trading_days(d, d)


def next_trading_day(date) -> pd.Timestamp:
    d = pd.Timestamp(date).normalize()
    span = trading_days(d + pd.Timedelta(days=1), d + pd.Timedelta(days=10))
    return span[0]


def previous_trading_day(date) -> pd.Timestamp:
    d = pd.Timestamp(date).normalize()
    span = trading_days(d - pd.Timedelta(days=10), d - pd.Timedelta(days=1))
    return span[-1]


def missing_sessions(
    price_index: pd.DatetimeIndex, start=None, end=None
) -> pd.DatetimeIndex:
    """Sessions the NYSE calendar expects but the price panel does not contain.

    These are candidate *data gaps* (as opposed to holidays, which the calendar
    already excludes). An empty result means the panel is session-complete.
    """
    idx = pd.DatetimeIndex(price_index).normalize().unique()
    if len(idx) == 0:
        return pd.DatetimeIndex([])
    expected = trading_days(start or idx.min(), end or idx.max())
    return expected.difference(idx)


def sessions_until_next_holiday(date, horizon: int = 10) -> int | None:
    """Trading sessions from `date` (exclusive) to the next full-day closure.

    Useful for a pre-holiday liquidity haircut (thinner books, wider spreads).
    Returns None if no closure falls within `horizon` calendar-weeks.
    """
    d = pd.Timestamp(date).normalize()
    future_hols = holidays(d + pd.Timedelta(days=1), d + pd.Timedelta(weeks=horizon))
    if len(future_hols) == 0:
        return None
    nxt = future_hols[0]
    return len(trading_days(d + pd.Timedelta(days=1), nxt - pd.Timedelta(days=1)))
