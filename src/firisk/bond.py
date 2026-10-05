"""Fixed-rate and zero-coupon bonds: cash flows, accrued interest, yield analytics.

Yield-based analytics use the U.S. street convention: the dirty price is the
cash flows discounted at the yield compounded ``freq`` times a year, with a
fractional first period ``w`` measured in the bond's day count::

    P_dirty = sum_k CF_k / (1 + y/f) ** (w + k),   k = 0..n-1

In the final coupon period (one cash flow left) the price uses simple
discounting, ``CF / (1 + w y / f)``, as SIFMA/Treasury conventions do.
Prices are per 100 face.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from functools import cached_property

import numpy as np
from scipy.optimize import brentq

from firisk.daycount import ACT_ACT, THIRTY_360, accrual_fraction, yf_act_365f
from firisk.schedule import bracketing_coupons, coupon_schedule

TSY = "TSY"
CORP = "CORP"


@dataclass(frozen=True)
class YieldAnalytics:
    yield_: float
    dirty: float
    clean: float
    accrued: float
    macaulay: float  # years
    modified: float  # years
    convexity: float  # years^2
    dv01: float  # price change per 100 face for a 1bp yield move (positive)


@dataclass(frozen=True)
class FixedRateBond:
    bond_id: str
    maturity: date
    coupon: float  # annual rate, decimal (0 for a zero-coupon bond)
    dated: date
    kind: str = TSY
    issuer: str = "U.S. Treasury"
    sector: str = "Government"
    rating: str = "AAA"
    freq: int = 2
    daycount: str = ACT_ACT
    eom: bool = True
    outstanding_bn: float = 1.0  # simulated amount outstanding, used for benchmark weights
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.daycount not in (ACT_ACT, THIRTY_360):
            raise ValueError(f"unsupported day count {self.daycount}")
        if self.coupon < 0:
            raise ValueError("coupon must be non-negative")

    @property
    def is_zero(self) -> bool:
        return self.coupon == 0.0

    @cached_property
    def schedule(self) -> list[date]:
        return coupon_schedule(self.dated, self.maturity, self.freq, self.eom)

    @property
    def coupon_amount(self) -> float:
        return 100.0 * self.coupon / self.freq

    def _check_settle(self, settle: date) -> None:
        if settle < self.dated:
            raise ValueError(f"{self.bond_id}: settlement {settle} before dated date {self.dated}")
        if settle >= self.maturity:
            raise ValueError(f"{self.bond_id}: settlement {settle} on/after maturity")

    def cashflows(self, settle: date) -> tuple[list[date], np.ndarray]:
        """Remaining cash flows strictly after ``settle`` (a coupon paid on the
        settlement date belongs to the seller)."""
        self._check_settle(settle)
        dates = [d for d in self.schedule[1:] if d > settle]
        amounts = np.full(len(dates), self.coupon_amount)
        amounts[-1] += 100.0
        return dates, amounts

    def cashflow_times(self, settle: date) -> tuple[np.ndarray, np.ndarray]:
        """Cash-flow times in ACT/365F years from settlement, and amounts."""
        dates, amounts = self.cashflows(settle)
        return np.array([yf_act_365f(settle, d) for d in dates]), amounts

    def accrued(self, settle: date) -> float:
        self._check_settle(settle)
        if self.is_zero:
            return 0.0
        prev, nxt = bracketing_coupons(self.schedule, settle)
        start = max(prev, self.dated)
        return self.coupon_amount * accrual_fraction(self.daycount, start, settle, prev, nxt,
                                                     self.freq)

    def first_period_fraction(self, settle: date) -> float:
        """w: fraction of a coupon period from settlement to the next schedule date."""
        prev, nxt = bracketing_coupons(self.schedule, settle)
        return accrual_fraction(self.daycount, settle, nxt, prev, nxt, self.freq)

    # ----- yield-based analytics -------------------------------------------------
    def yield_analytics(self, y: float, settle: date) -> YieldAnalytics:
        _, cf = self.cashflows(settle)
        f = self.freq
        w = self.first_period_fraction(settle)
        acc = self.accrued(settle)
        if len(cf) == 1:  # final period: simple (money-market) discounting
            tau = w / f
            g = 1.0 + y * tau
            dirty = cf[0] / g
            d1 = -cf[0] * tau / g**2
            d2 = 2.0 * cf[0] * tau**2 / g**3
            mac = tau
        else:
            n = w + np.arange(len(cf))
            v = 1.0 / (1.0 + y / f)
            pv = cf * v**n
            dirty = pv.sum()
            d1 = -(cf * (n / f) * v ** (n + 1)).sum()
            d2 = (cf * (n * (n + 1) / f**2) * v ** (n + 2)).sum()
            mac = ((n / f) * pv).sum() / dirty
        modified = -d1 / dirty
        return YieldAnalytics(
            yield_=y, dirty=dirty, clean=dirty - acc, accrued=acc, macaulay=mac,
            modified=modified, convexity=d2 / dirty, dv01=modified * dirty * 1e-4,
        )

    def dirty_from_yield(self, y: float, settle: date) -> float:
        return self.yield_analytics(y, settle).dirty

    def clean_from_yield(self, y: float, settle: date) -> float:
        return self.yield_analytics(y, settle).clean

    def ytm(self, price: float, settle: date, clean: bool = True) -> float:
        """Yield to maturity from a clean (default) or dirty price."""
        target = price + (self.accrued(settle) if clean else 0.0)
        return brentq(lambda y: self.dirty_from_yield(y, settle) - target, -0.05, 1.0,
                      xtol=1e-14, rtol=1e-14, maxiter=200)
