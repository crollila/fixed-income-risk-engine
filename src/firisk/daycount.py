"""Day-count conventions used by U.S. Treasuries and corporate bonds.

* Treasuries accrue on Actual/Actual (ICMA): accrued days divided by the actual
  days in the coupon period.
* U.S. corporates accrue on 30/360 (SIFMA "30/360 US", a.k.a. Bond Basis with
  the February end-of-month rule).
* Curve time is measured in Actual/365 Fixed years from the valuation date.
"""

from __future__ import annotations

import calendar
from datetime import date

ACT_ACT = "ACT/ACT"
THIRTY_360 = "30/360"


def is_month_end(d: date) -> bool:
    return d.day == calendar.monthrange(d.year, d.month)[1]


def _is_last_day_of_feb(d: date) -> bool:
    return d.month == 2 and is_month_end(d)


def days_30_360_us(d1: date, d2: date) -> int:
    """30/360 US day count (SIFMA), including the February end-of-month rules."""
    y1, m1, dd1 = d1.year, d1.month, d1.day
    y2, m2, dd2 = d2.year, d2.month, d2.day
    if _is_last_day_of_feb(d1) and _is_last_day_of_feb(d2):
        dd2 = 30
    if _is_last_day_of_feb(d1):
        dd1 = 30
    if dd2 == 31 and dd1 >= 30:
        dd2 = 30
    if dd1 == 31:
        dd1 = 30
    return 360 * (y2 - y1) + 30 * (m2 - m1) + (dd2 - dd1)


def act_days(d1: date, d2: date) -> int:
    return (d2 - d1).days


def yf_act_365f(d1: date, d2: date) -> float:
    return act_days(d1, d2) / 365.0


def yf_act_360(d1: date, d2: date) -> float:
    return act_days(d1, d2) / 360.0


def accrual_fraction(convention: str, start: date, end: date, period_start: date,
                     period_end: date, freq: int) -> float:
    """Fraction of one regular coupon covered by [start, end] inside a coupon period.

    ACT/ACT (ICMA): actual days / actual days in the period.
    30/360 US: 30/360 days / (360 / freq), so a full regular period is always one coupon.
    """
    if convention == ACT_ACT:
        return act_days(start, end) / act_days(period_start, period_end)
    if convention == THIRTY_360:
        return days_30_360_us(start, end) / (360.0 / freq)
    raise ValueError(f"unknown day count {convention!r}")
