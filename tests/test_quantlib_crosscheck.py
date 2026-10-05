"""Optional independent cross-check against QuantLib (skipped if it is not installed).

QuantLib is used only here, never by the engine itself.
"""

from datetime import date

import pytest

from firisk.bond import FixedRateBond
from firisk.daycount import ACT_ACT
from firisk.universe import UNIVERSE

ql = pytest.importorskip("QuantLib")

VAL = date(2026, 10, 1)


def _qd(d: date):
    return ql.Date(d.day, d.month, d.year)


def _ql_bond(b: FixedRateBond):
    sched = ql.Schedule(_qd(b.dated), _qd(b.maturity), ql.Period(ql.Semiannual), ql.NullCalendar(),
                        ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Backward, b.eom)
    if b.daycount == ACT_ACT:
        dc = ql.ActualActual(ql.ActualActual.Bond, sched)
    else:
        dc = ql.Thirty360(ql.Thirty360.USA)
    return ql.FixedRateBond(0, 100.0, sched, [b.coupon], dc, ql.Unadjusted), dc


@pytest.mark.parametrize("bond_id", ["UST 5.250 08/15/36", "UST 1.375 11/15/31", "UST 5.625 08/15/56",
                                     "ATLAS 5.875 03/15/36", "SKYL 6.650 12/01/54", "IRON 5.750 09/15/31"])
def test_price_accrued_and_duration_match_quantlib(bond_id):
    ql.Settings.instance().evaluationDate = _qd(VAL)
    b = next(x for x in UNIVERSE if x.bond_id == bond_id)
    qb, dc = _ql_bond(b)
    y = 0.0537
    rate = ql.InterestRate(y, dc, ql.Compounded, ql.Semiannual)
    ours = b.yield_analytics(y, VAL)
    assert ours.accrued == pytest.approx(qb.accruedAmount(_qd(VAL)), abs=1e-9)
    assert ours.clean == pytest.approx(ql.BondFunctions.cleanPrice(qb, rate, _qd(VAL)), abs=1e-8)
    assert ours.modified == pytest.approx(ql.BondFunctions.duration(qb, rate, ql.Duration.Modified, _qd(VAL)), rel=1e-8)
    assert ours.macaulay == pytest.approx(ql.BondFunctions.duration(qb, rate, ql.Duration.Macaulay, _qd(VAL)), rel=1e-8)
    assert ours.convexity == pytest.approx(ql.BondFunctions.convexity(qb, rate, _qd(VAL)), rel=1e-6)
    assert b.ytm(ours.clean, VAL) == pytest.approx(
        ql.BondFunctions.bondYield(qb, ql.BondPrice(ours.clean, ql.BondPrice.Clean), dc, ql.Compounded,
                                   ql.Semiannual, _qd(VAL)), abs=1e-9)
