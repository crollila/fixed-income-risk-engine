"""Bootstrap a zero curve from Treasury constant-maturity par yields.

Instruments (a stylized year-fraction grid, documented simplification):

* tenors <= 6M are treated as bills: ``DF = 1 / (1 + y t)`` (bond-equivalent,
  simple interest);
* tenors >= 1Y are par bonds paying ``y / 2`` semiannually at t = 0.5, 1.0, ...,
  T, priced at exactly 100.

Pillars are solved one at a time. Coupon dates that fall between two pillars
are discounted with the curve's own log-linear interpolation, so the solved
curve reprices every input instrument to par (to ~1e-12).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from firisk.curve import ZeroCurve

BILL_MAX_T = 0.5


def bootstrap_par_curve(tenors: np.ndarray, par_yields: np.ndarray) -> ZeroCurve:
    tenors = np.asarray(tenors, dtype=float)
    par = np.asarray(par_yields, dtype=float)
    if np.any(np.diff(tenors) <= 0):
        raise ValueError("tenors must be strictly increasing")
    times: list[float] = []
    zeros: list[float] = []
    for t, y in zip(tenors, par, strict=True):
        if t <= BILL_MAX_T + 1e-12:
            times.append(t)
            zeros.append(np.log1p(y * t) / t)
            continue
        if not times or times[-1] < BILL_MAX_T - 1e-12:
            raise ValueError("coupon pillars need a 6M pillar for the first coupon")
        coupon_times = np.arange(1, int(round(2 * t)) + 1) * 0.5
        cpn = 100.0 * y / 2.0

        def par_error(z: float, t=t, coupon_times=coupon_times, cpn=cpn) -> float:
            curve = ZeroCurve(np.array(times + [t]), np.array(zeros + [z]))
            dfs = curve.df(coupon_times)
            return cpn * dfs.sum() + 100.0 * dfs[-1] - 100.0

        z = brentq(par_error, -0.2, 0.5, xtol=1e-15, rtol=4 * np.finfo(float).eps, maxiter=200)
        times.append(t)
        zeros.append(z)
    return ZeroCurve(np.array(times), np.array(zeros))


def par_instrument_prices(curve: ZeroCurve, tenors: np.ndarray, par_yields: np.ndarray) -> np.ndarray:
    """Reprice the bootstrap instruments on ``curve`` (100 means exact fit)."""
    out = []
    for t, y in zip(tenors, par_yields, strict=True):
        if t <= BILL_MAX_T + 1e-12:
            out.append(100.0 * curve.df(t) * (1.0 + y * t))
        else:
            ct = np.arange(1, int(round(2 * t)) + 1) * 0.5
            dfs = curve.df(ct)
            out.append(100.0 * y / 2.0 * dfs.sum() + 100.0 * dfs[-1])
    return np.array(out, dtype=float)


def par_yield(curve: ZeroCurve, t: float) -> float:
    """Model par yield (semiannual coupon) for maturity t on a 0.5y grid."""
    ct = np.arange(1, int(round(2 * t)) + 1) * 0.5
    dfs = curve.df(ct)
    return float(2.0 * (1.0 - dfs[-1]) / dfs.sum())
