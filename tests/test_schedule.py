from datetime import date

import pytest

from firisk.schedule import add_months, bracketing_coupons, coupon_schedule


def test_add_months_end_of_month_rule():
    assert add_months(date(2025, 8, 31), -6) == date(2025, 2, 28)
    assert add_months(date(2023, 8, 31), -6) == date(2023, 2, 28)
    assert add_months(date(2024, 2, 29), 6) == date(2024, 8, 31)
    assert add_months(date(2024, 2, 29), 6, eom=False) == date(2024, 8, 29)
    assert add_months(date(2025, 5, 15), -6) == date(2024, 11, 15)


def test_treasury_month_end_schedule_with_leap_february():
    s = coupon_schedule(date(2026, 2, 28), date(2028, 2, 29), 2)
    assert s == [date(2026, 2, 28), date(2026, 8, 31), date(2027, 2, 28), date(2027, 8, 31),
                 date(2028, 2, 29)]


def test_mid_month_schedule_and_off_cycle_dated_date():
    s = coupon_schedule(date(2025, 3, 1), date(2027, 5, 15), 2)
    assert s[0] == date(2024, 11, 15)  # period containing the dated date
    assert s[-1] == date(2027, 5, 15)
    assert all(d.day == 15 for d in s)
    assert len(s) == 6


def test_bracketing_coupons_on_and_between_dates():
    s = coupon_schedule(date(2025, 2, 15), date(2030, 2, 15), 2)
    assert bracketing_coupons(s, date(2026, 8, 15)) == (date(2026, 8, 15), date(2027, 2, 15))
    assert bracketing_coupons(s, date(2026, 8, 14)) == (date(2026, 2, 15), date(2026, 8, 15))
    with pytest.raises(ValueError):
        bracketing_coupons(s, date(2030, 2, 15))
    with pytest.raises(ValueError):
        coupon_schedule(date(2030, 1, 1), date(2029, 1, 1), 2)
