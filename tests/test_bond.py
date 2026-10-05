"""Bond analytics against independent closed-form fixtures."""

from datetime import date

import numpy as np
import pytest

from firisk.bond import CORP, FixedRateBond
from firisk.curve import ZeroCurve
from firisk.daycount import THIRTY_360, days_30_360_us
from firisk.market import BondBook
from firisk.universe import UNIVERSE

SETTLE = date(2026, 2, 15)
VAL = date(2026, 10, 1)


def bullet(coupon, years, settle=SETTLE, **kw):
    return FixedRateBond("T", date(settle.year + years, settle.month, settle.day), coupon, settle, **kw)


def annuity_price(c, y, n_periods):
    """Closed-form price of a bond on a coupon date (independent of the engine)."""
    r = y / 2
    return 100 * (c / 2) / r * (1 - (1 + r) ** -n_periods) + 100 * (1 + r) ** -n_periods


def test_textbook_discount_bond_price():
    # 10-year 6% semiannual bond at an 8% yield: 86.4097 (standard textbook example)
    b = bullet(0.06, 10)
    assert b.dirty_from_yield(0.08, SETTLE) == pytest.approx(86.409674, abs=1e-6)
    assert b.dirty_from_yield(0.08, SETTLE) == pytest.approx(annuity_price(0.06, 0.08, 20), abs=1e-10)


@pytest.mark.parametrize("y", [0.01, 0.045, 0.08])
def test_par_bond_prices_at_par_and_closed_form_durations(y):
    b = bullet(y, 7)
    a = b.yield_analytics(y, SETTLE)
    n = 14
    assert a.dirty == pytest.approx(100.0, abs=1e-10)
    assert a.accrued == 0.0
    assert a.macaulay == pytest.approx((1 + y / 2) / y * (1 - (1 + y / 2) ** -n), rel=1e-12)
    assert a.modified == pytest.approx(1 / y * (1 - (1 + y / 2) ** -n), rel=1e-12)


def test_zero_coupon_identity_on_and_off_cycle():
    z = FixedRateBond("Z", date(2036, 8, 15), 0.0, date(2026, 2, 15))
    y = 0.05
    assert z.dirty_from_yield(y, SETTLE) == pytest.approx(100 / (1 + y / 2) ** 21, rel=1e-13)
    settle = date(2026, 5, 15)
    w = (date(2026, 8, 15) - settle).days / (date(2026, 8, 15) - date(2026, 2, 15)).days
    assert z.dirty_from_yield(y, settle) == pytest.approx(100 / (1 + y / 2) ** (w + 20), rel=1e-13)
    assert z.accrued(settle) == 0.0
    a = z.yield_analytics(y, settle)
    assert a.macaulay == pytest.approx((w + 20) / 2, rel=1e-12)  # duration of a zero = its maturity


def test_zero_coupon_under_curve_is_discount_factor():
    z = FixedRateBond("Z", date(2036, 2, 15), 0.0, date(2026, 2, 15))
    curve = ZeroCurve(np.array([1.0, 5.0, 30.0]), np.array([0.04, 0.045, 0.05]))
    book = BondBook([z], SETTLE)
    t = (date(2036, 2, 15) - SETTLE).days / 365
    assert book.dirty(curve, 0.0)[0] == pytest.approx(100 * float(curve.df(t)), rel=1e-14)


@pytest.mark.parametrize("bond", UNIVERSE, ids=lambda b: b.bond_id)
def test_price_strictly_decreases_with_yield(bond):
    ys = np.linspace(0.0, 0.15, 61)
    p = np.array([bond.dirty_from_yield(y, VAL) for y in ys])
    assert np.all(np.diff(p) < 0)


def test_ytm_round_trip_clean_and_dirty():
    b = UNIVERSE[5]
    clean = b.clean_from_yield(0.0512345, VAL)
    assert b.ytm(clean, VAL) == pytest.approx(0.0512345, abs=1e-12)
    assert b.ytm(clean + b.accrued(VAL), VAL, clean=False) == pytest.approx(0.0512345, abs=1e-12)


@pytest.mark.parametrize("bond", [UNIVERSE[1], UNIVERSE[10], UNIVERSE[8], UNIVERSE[20]], ids=lambda b: b.bond_id)
def test_analytic_dv01_and_convexity_match_finite_differences(bond):
    y, h = 0.055, 1e-4
    a = bond.yield_analytics(y, VAL)
    up, dn = bond.dirty_from_yield(y + h, VAL), bond.dirty_from_yield(y - h, VAL)
    assert a.dv01 == pytest.approx((dn - up) / 2, rel=1e-6)
    h2 = 1e-3
    up2, dn2 = bond.dirty_from_yield(y + h2, VAL), bond.dirty_from_yield(y - h2, VAL)
    assert a.convexity == pytest.approx((up2 + dn2 - 2 * a.dirty) / (a.dirty * h2**2), rel=1e-4)


def test_settlement_on_coupon_date_excludes_that_coupon():
    b = FixedRateBond("T", date(2030, 8, 15), 0.05, date(2025, 8, 15))
    on = date(2026, 8, 15)
    dates, cf = b.cashflows(on)
    assert dates[0] == date(2027, 2, 15)
    assert b.accrued(on) == 0.0
    before = date(2026, 8, 14)
    assert b.cashflows(before)[0][0] == on
    assert b.accrued(before) == pytest.approx(2.5 * 180 / 181)
    assert len(cf) == 8 and cf[-1] == pytest.approx(102.5)


def test_30_360_accrued_with_day_31_and_clean_dirty_identity():
    b = FixedRateBond("C", date(2031, 3, 15), 0.06, date(2025, 3, 15), kind=CORP, daycount=THIRTY_360)
    settle = date(2025, 10, 31)
    assert days_30_360_us(date(2025, 9, 15), settle) == 46  # D2=31 kept because D1<30
    assert b.accrued(settle) == pytest.approx(100 * 0.06 * 46 / 360)
    a = b.yield_analytics(0.065, settle)
    assert a.clean == pytest.approx(a.dirty - a.accrued, abs=1e-12)


def test_final_period_uses_simple_yield():
    b = FixedRateBond("T", date(2027, 3, 31), 0.0425, date(2025, 3, 31))
    w = (date(2027, 3, 31) - VAL).days / (date(2027, 3, 31) - date(2026, 9, 30)).days
    y = 0.043
    assert b.dirty_from_yield(y, VAL) == pytest.approx(102.125 / (1 + w * y / 2), rel=1e-14)
    a = b.yield_analytics(y, VAL)
    h = 1e-5
    fd = (b.dirty_from_yield(y - h, VAL) - b.dirty_from_yield(y + h, VAL)) / 2
    assert a.dv01 == pytest.approx(fd * 1e-4 / h, rel=1e-8)


def test_invalid_settlement_raises():
    b = FixedRateBond("T", date(2030, 8, 15), 0.05, date(2025, 8, 15))
    with pytest.raises(ValueError):
        b.accrued(date(2025, 1, 1))
    with pytest.raises(ValueError):
        b.cashflows(date(2030, 8, 15))
    with pytest.raises(ValueError):
        FixedRateBond("X", date(2030, 1, 1), -0.01, date(2025, 1, 1))
