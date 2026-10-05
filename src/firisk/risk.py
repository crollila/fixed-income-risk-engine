"""Bond-level sensitivities by full repricing on bumped curves.

Conventions (all per 100 face; multiply by face/100 for a position):

* DV01   = -(P(+1bp) - P(-1bp)) / 2 for a parallel 1bp shift of every par-curve
           input, with the zero curve re-bootstrapped (positive = loses value
           when rates rise).
* Effective duration  = DV01 / (P * 1e-4).
* Effective convexity = (P(+h) + P(-h) - 2P) / (P h^2), h = 10bp.
* Key-rate DV01_k = central 1bp bump of par tenor k only, re-bootstrapped.
  The bumps sum to the parallel bump, so key-rate DV01s sum to DV01 up to
  second-order (bootstrap non-linearity) terms.
* CS01   = -(P(z+1bp) - P(z-1bp)) / 2 bumping the bond's z-spread. Defined for
           credit (CORP) instruments only; Treasuries are the risk-free reference
           and carry zero credit exposure by construction.
* Spread duration = CS01 / (P * 1e-4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from firisk.bond import CORP, FixedRateBond
from firisk.market import BondBook, Market

BP = 1e-4
CONVEXITY_STEP_BP = 10.0


@dataclass
class RiskTable:
    frame: pd.DataFrame  # one row per bond, per-100-face measures
    kr_dv01: np.ndarray  # (n_bonds, n_tenors) per 100 face
    tenor_labels: list[str]


def bond_risk(bonds: list[FixedRateBond], market: Market, z: np.ndarray,
              tenor_labels: list[str]) -> RiskTable:
    book = BondBook(bonds, market.valuation_date)
    z = np.asarray(z, dtype=float)
    p0 = book.dirty(market.curve, z)

    up, dn = market.shifted(1.0), market.shifted(-1.0)
    dv01 = -(book.dirty(up.curve, z) - book.dirty(dn.curve, z)) / 2.0
    h = CONVEXITY_STEP_BP
    pu, pd_ = book.dirty(market.shifted(h).curve, z), book.dirty(market.shifted(-h).curve, z)
    eff_conv = (pu + pd_ - 2 * p0) / (p0 * (h * BP) ** 2)

    n_t = len(market.tenors)
    kr = np.empty((len(bonds), n_t))
    for k in range(n_t):
        e = np.zeros(n_t)
        e[k] = 1.0
        kr[:, k] = -(book.dirty(market.shifted(e).curve, z) - book.dirty(market.shifted(-e).curve, z)) / 2.0

    is_credit = np.array([b.kind == CORP for b in bonds])
    cs01_raw = -(book.dirty(market.curve, z + BP) - book.dirty(market.curve, z - BP)) / 2.0
    cs01 = np.where(is_credit, cs01_raw, 0.0)

    rows = []
    for i, b in enumerate(bonds):
        clean = p0[i] - book.accrued[i]
        y = b.ytm(p0[i], market.valuation_date, clean=False)
        ya = b.yield_analytics(y, market.valuation_date)
        rows.append({
            "bond_id": b.bond_id, "issuer": b.issuer, "sector": b.sector, "rating": b.rating,
            "kind": b.kind, "coupon": b.coupon, "maturity": b.maturity.isoformat(),
            "years_to_maturity": (b.maturity - market.valuation_date).days / 365.25,
            "clean": clean, "accrued": book.accrued[i], "dirty": p0[i], "ytm": y,
            "z_spread_bp": z[i] / BP if b.kind == CORP else 0.0,
            "macaulay": ya.macaulay, "modified": ya.modified, "yield_convexity": ya.convexity,
            "yield_dv01": ya.dv01, "eff_duration": dv01[i] / (p0[i] * BP), "eff_convexity": eff_conv[i],
            "dv01": dv01[i], "kr_dv01_sum": kr[i].sum(), "cs01": cs01[i],
            "spread_duration": cs01[i] / (p0[i] * BP),
        })
    return RiskTable(pd.DataFrame(rows), kr, list(tenor_labels))
