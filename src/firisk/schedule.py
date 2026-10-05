"""Coupon schedules (unadjusted dates, generated backward from maturity)."""

from __future__ import annotations

import calendar
from bisect import bisect_right
from datetime import date

from firisk.daycount import is_month_end


def add_months(d: date, months: int, eom: bool = True) -> date:
    """Shift by whole months; with ``eom`` a month-end date stays a month-end date."""
    total = d.year * 12 + (d.month - 1) + months
    y, m = divmod(total, 12)
    m += 1
    last = calendar.monthrange(y, m)[1]
    day = last if (eom and is_month_end(d)) else min(d.day, last)
    return date(y, m, day)


def coupon_schedule(dated: date, maturity: date, freq: int, eom: bool = True) -> list[date]:
    """All schedule dates from the first period start (<= dated date) to maturity.

    Dates are generated backward from maturity by ``12 / freq`` months, each
    computed directly from maturity so day-of-month never drifts. The first
    element is the start of the period containing the dated date; for a bond
    dated on-cycle it equals the dated date.
    """
    if maturity <= dated:
        raise ValueError("maturity must be after the dated date")
    if 12 % freq:
        raise ValueError("frequency must divide 12")
    step = 12 // freq
    dates = [maturity]
    k = 1
    while dates[-1] > dated:
        dates.append(add_months(maturity, -k * step, eom))
        k += 1
    return dates[::-1]


def bracketing_coupons(schedule: list[date], settle: date) -> tuple[date, date]:
    """(previous, next) schedule dates with previous <= settle < next."""
    if settle < schedule[0] or settle >= schedule[-1]:
        raise ValueError(f"settlement {settle} outside schedule {schedule[0]}..{schedule[-1]}")
    i = bisect_right(schedule, settle)
    return schedule[i - 1], schedule[i]
