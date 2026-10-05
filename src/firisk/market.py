"""Market state (par curve -> bootstrapped zero curve) and vectorized curve pricing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import cached_property

import numpy as np
from scipy.optimize import brentq

from firisk.bond import FixedRateBond
from firisk.bootstrap import bootstrap_par_curve
from firisk.curve import ZeroCurve


@dataclass(frozen=True)
class Market:
    valuation_date: date
    tenors: np.ndarray  # years
    par_yields: np.ndarray  # decimal, bond-equivalent

    @cached_property
    def curve(self) -> ZeroCurve:
        return bootstrap_par_curve(self.tenors, self.par_yields)

    def shifted(self, par_shift_bp) -> Market:
        """New market with par yields shifted by ``par_shift_bp`` (scalar or per tenor)."""
        shift = np.broadcast_to(np.asarray(par_shift_bp, dtype=float), self.par_yields.shape)
        return Market(self.valuation_date, self.tenors, self.par_yields + shift * 1e-4)


class BondBook:
    """Cash flows of many bonds stacked for one-shot vectorized curve pricing.

    Settlement equals the valuation date (T+0, a documented simplification).
    Dirty price per 100 face: ``sum CF * DF(t) * exp(-z t)`` with ``z`` the
    bond's z-spread (continuously compounded) over the Treasury zero curve.
    """

    def __init__(self, bonds: list[FixedRateBond], settle: date):
        self.bonds = list(bonds)
        self.settle = settle
        times, amts, idx = [], [], []
        for i, b in enumerate(self.bonds):
            t, a = b.cashflow_times(settle)
            times.append(t)
            amts.append(a)
            idx.append(np.full(t.size, i))
        self.t = np.concatenate(times)
        self.cf = np.concatenate(amts)
        self.idx = np.concatenate(idx)
        self.accrued = np.array([b.accrued(settle) for b in self.bonds])

    def __len__(self) -> int:
        return len(self.bonds)

    def dirty(self, curve: ZeroCurve, z: np.ndarray) -> np.ndarray:
        z = np.broadcast_to(np.asarray(z, dtype=float), (len(self),))
        pv = self.cf * curve.df(self.t) * np.exp(-z[self.idx] * self.t)
        return np.bincount(self.idx, weights=pv, minlength=len(self))

    def z_spreads(self, curve: ZeroCurve, dirty: np.ndarray) -> np.ndarray:
        out = np.empty(len(self))
        for i in range(len(self)):
            m = self.idx == i
            t, cf, dfs = self.t[m], self.cf[m], curve.df(self.t[m])
            out[i] = brentq(lambda z, t=t, cf=cf, dfs=dfs, p=dirty[i]: (cf * dfs * np.exp(-z * t)).sum() - p,
                            -0.5, 1.0, xtol=1e-15, rtol=4 * np.finfo(float).eps)
        return out
