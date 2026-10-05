from datetime import date

import pytest

from firisk.daycount import ACT_ACT, THIRTY_360, accrual_fraction, days_30_360_us, yf_act_360, yf_act_365f


@pytest.mark.parametrize("d1,d2,expected", [
    (date(2025, 1, 15), date(2025, 7, 15), 180),   # regular half year
    (date(2025, 1, 31), date(2025, 2, 28), 28),    # D1=31 -> 30
    (date(2025, 1, 30), date(2025, 3, 31), 60),    # D2=31 and D1>=30 -> 30
    (date(2025, 1, 15), date(2025, 3, 31), 76),    # D2=31 kept when D1<30
    (date(2025, 2, 28), date(2025, 3, 31), 30),    # last day of Feb -> 30, then D2 31 -> 30
    (date(2025, 2, 28), date(2025, 8, 31), 180),   # EOM semiannual period
    (date(2024, 2, 29), date(2025, 2, 28), 360),   # both last-of-Feb (leap -> non-leap)
    (date(2024, 2, 28), date(2024, 3, 1), 3),      # Feb 28 of a leap year is not month end
])
def test_30_360_us_cases(d1, d2, expected):
    assert days_30_360_us(d1, d2) == expected


def test_act_fractions():
    assert yf_act_365f(date(2025, 1, 1), date(2026, 1, 1)) == 1.0
    assert yf_act_360(date(2025, 1, 1), date(2025, 1, 31)) == pytest.approx(30 / 360)
    # ACT/ACT ICMA: 46 of 181 days into a Feb 15 - Aug 15 period
    f = accrual_fraction(ACT_ACT, date(2026, 2, 15), date(2026, 4, 2), date(2026, 2, 15), date(2026, 8, 15), 2)
    assert f == pytest.approx(46 / 181)


def test_30_360_fraction_is_per_regular_coupon():
    # A full 30/360 period is exactly one coupon even when the period spans February.
    assert accrual_fraction(THIRTY_360, date(2025, 8, 31), date(2026, 2, 28), date(2025, 8, 31),
                            date(2026, 2, 28), 2) == pytest.approx(178 / 180)
    assert accrual_fraction(THIRTY_360, date(2025, 3, 15), date(2025, 9, 15), date(2025, 3, 15),
                            date(2025, 9, 15), 2) == 1.0
    with pytest.raises(ValueError):
        accrual_fraction("ACT/999", date(2025, 1, 1), date(2025, 2, 1), date(2025, 1, 1), date(2025, 7, 1), 2)
