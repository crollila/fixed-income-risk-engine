"""Instantaneous stress scenarios: full repricing vs. sensitivity approximations.

Rate shocks move the Treasury *par* curve tenor by tenor; the zero curve is
re-bootstrapped and every bond fully repriced. Spread shocks move corporate
z-spreads. Approximations use the same sensitivities the risk report shows:

* first order : -sum_k KRDV01_k * dy_k  -  CS01 * ds
* second order: first order + 1/2 * Convexity * MV * (dr + ds)^2, where dr is
  the bond's KR-DV01-weighted average rate shift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from firisk.bond import CORP, FixedRateBond
from firisk.market import BondBook, Market
from firisk.risk import RiskTable


@dataclass(frozen=True)
class Scenario:
    name: str
    rate_bp: np.ndarray  # per par tenor
    spread_bp: dict[str, float] = field(default_factory=dict)  # by rating, corporates only
    family: str = "rates"


def parallel(bp: float, tenors: np.ndarray) -> np.ndarray:
    return np.full(len(tenors), float(bp))


def twist(short_bp: float, long_bp: float, tenors: np.ndarray, t_short: float = 2.0,
          t_long: float = 10.0) -> np.ndarray:
    """``short_bp`` at/below t_short, ``long_bp`` at/above t_long, linear in between."""
    return np.interp(tenors, [t_short, t_long], [short_bp, long_bp])


CREDIT_WIDENING = {"AA": 30.0, "A": 50.0, "BBB": 80.0}


def standard_scenarios(tenors: np.ndarray) -> list[Scenario]:
    s = [Scenario(f"Parallel {b:+d}bp", parallel(b, tenors)) for b in (25, 50, 100, -25, -50, -100)]
    s += [
        Scenario("Bear steepener (2s -25 / 10s+ +25)", twist(-25, 25, tenors), family="curve"),
        Scenario("Bull flattener (2s +25 / 10s+ -25)", twist(25, -25, tenors), family="curve"),
        Scenario("IG spreads widen (AA+30/A+50/BBB+80)", parallel(0, tenors), CREDIT_WIDENING, "credit"),
        Scenario("Risk-off: rates -50 + IG widening", parallel(-50, tenors), CREDIT_WIDENING, "combined"),
        Scenario("Stagflation: rates +100 + IG widening", parallel(100, tenors), CREDIT_WIDENING, "combined"),
    ]
    return s


class StressEngine:
    def __init__(self, bonds: list[FixedRateBond], market: Market, z: np.ndarray, risk: RiskTable):
        self.bonds, self.market, self.z, self.risk = bonds, market, np.asarray(z), risk
        self.book = BondBook(bonds, market.valuation_date)
        self.p0 = self.book.dirty(market.curve, self.z)
        self.is_credit = np.array([b.kind == CORP for b in bonds])
        self.ratings = np.array([b.rating for b in bonds])

    def spread_vector_bp(self, sc: Scenario) -> np.ndarray:
        return np.array([sc.spread_bp.get(r, 0.0) if c else 0.0
                         for r, c in zip(self.ratings, self.is_credit, strict=True)])

    def full_pnl(self, face: np.ndarray, rate_bp: np.ndarray, ds_bp: np.ndarray) -> np.ndarray:
        curve = self.market.shifted(rate_bp).curve if np.any(rate_bp) else self.market.curve
        p1 = self.book.dirty(curve, self.z + ds_bp * 1e-4)
        return face / 100.0 * (p1 - self.p0)

    def approx_pnl(self, face: np.ndarray, rate_bp: np.ndarray, ds_bp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f = self.risk.frame
        scale = face / 100.0
        kr_move = self.risk.kr_dv01 @ rate_bp  # per 100 face
        first = -scale * (kr_move + f["cs01"].to_numpy() * ds_bp)
        dv01 = f["dv01"].to_numpy()
        dr = np.divide(kr_move, dv01, out=np.zeros_like(kr_move), where=dv01 != 0)
        mv = scale * self.p0
        second = first + 0.5 * f["eff_convexity"].to_numpy() * mv * ((dr + ds_bp) * 1e-4) ** 2
        return first, second

    def run(self, face: np.ndarray, sc: Scenario) -> dict:
        ds = self.spread_vector_bp(sc)
        zero = np.zeros_like(sc.rate_bp)
        total = self.full_pnl(face, sc.rate_bp, ds).sum()
        rates_only = self.full_pnl(face, sc.rate_bp, np.zeros_like(ds)).sum()
        spread_only = self.full_pnl(face, zero, ds).sum()
        first, second = self.approx_pnl(face, sc.rate_bp, ds)
        return {
            "scenario": sc.name, "family": sc.family,
            "full_pnl": total, "rates_pnl": rates_only, "spread_pnl": spread_only,
            "interaction_pnl": total - rates_only - spread_only,
            "first_order_pnl": first.sum(), "second_order_pnl": second.sum(),
            "first_order_error": first.sum() - total, "second_order_error": second.sum() - total,
        }

    def run_all(self, face: np.ndarray, scenarios: list[Scenario], nav: float) -> pd.DataFrame:
        df = pd.DataFrame([self.run(face, s) for s in scenarios])
        df["full_pnl_pct_nav"] = df["full_pnl"] / nav
        return df

    def convergence(self, face: np.ndarray, shocks_bp: list[float]) -> pd.DataFrame:
        rows = []
        n = len(self.market.tenors)
        zero_ds = np.zeros(len(self.bonds))
        for b in shocks_bp:
            r = np.full(n, float(b))
            full = self.full_pnl(face, r, zero_ds).sum()
            first, second = self.approx_pnl(face, r, zero_ds)
            rows.append({"shock_bp": b, "full_pnl": full, "first_order_pnl": first.sum(),
                         "second_order_pnl": second.sum(),
                         "abs_err_first": abs(first.sum() - full), "abs_err_second": abs(second.sum() - full)})
        return pd.DataFrame(rows)
