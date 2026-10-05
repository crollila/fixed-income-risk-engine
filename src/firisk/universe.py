"""SIMULATED instrument universe, portfolio and benchmark.

Everything in this module is manufactured for demonstration:

* Treasury lines use real U.S. Treasury conventions (semiannual, ACT/ACT,
  month-end / 15th maturities) but are synthetic securities - no CUSIPs.
* Corporate issuers are fictional names. Ratings, coupons, z-spreads,
  amounts outstanding and transaction costs are synthetic assumptions.
* The portfolio is a simulated ~$100MM book. It is not real or client AUM.

Only the Treasury par curve (FRED H.15 cache) is market data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from firisk.bond import CORP, TSY, FixedRateBond
from firisk.daycount import ACT_ACT, THIRTY_360

SIMULATED_NAV = 100_000_000.0  # simulated portfolio size; NOT real AUM
FACE_ROUNDING = 1_000.0


def _tsy(bid, cpn, mat, dated, out_bn):
    return FixedRateBond(bid, mat, cpn, dated, kind=TSY, daycount=ACT_ACT, outstanding_bn=out_bn)


def _corp(bid, issuer, sector, rating, cpn, mat, dated, out_bn):
    return FixedRateBond(bid, mat, cpn, dated, kind=CORP, issuer=issuer, sector=sector,
                         rating=rating, daycount=THIRTY_360, outstanding_bn=out_bn)


D = date
TREASURIES: list[FixedRateBond] = [
    _tsy("UST 4.250 03/31/27", 0.04250, D(2027, 3, 31), D(2025, 3, 31), 70),
    _tsy("UST 4.625 09/30/28", 0.04625, D(2028, 9, 30), D(2026, 9, 30), 69),
    _tsy("UST 4.875 09/15/29", 0.04875, D(2029, 9, 15), D(2026, 9, 15), 58),
    _tsy("UST 4.250 01/31/30", 0.04250, D(2030, 1, 31), D(2025, 1, 31), 70),
    _tsy("UST 5.000 09/30/31", 0.05000, D(2031, 9, 30), D(2026, 9, 30), 70),
    _tsy("UST 1.375 11/15/31", 0.01375, D(2031, 11, 15), D(2021, 11, 15), 61),
    _tsy("UST 5.125 09/30/33", 0.05125, D(2033, 9, 30), D(2026, 9, 30), 44),
    _tsy("UST 5.250 08/15/36", 0.05250, D(2036, 8, 15), D(2026, 8, 15), 42),
    _tsy("UST STRIP 0 11/15/36", 0.0, D(2036, 11, 15), D(2026, 5, 15), 8),
    _tsy("UST 4.750 02/15/45", 0.04750, D(2045, 2, 15), D(2025, 2, 15), 16),
    _tsy("UST 5.625 08/15/56", 0.05625, D(2056, 8, 15), D(2026, 8, 15), 25),
]

FIN, TECH, HC, ENER, UTIL, IND, STAP, COMM = (
    "Financials", "Technology", "Healthcare", "Energy", "Utilities", "Industrials",
    "Consumer Staples", "Communications",
)
CORPORATES: list[FixedRateBond] = [
    _corp("ATLAS 4.200 01/15/29", "Atlas Bancorp", FIN, "A", 0.0420, D(2029, 1, 15), D(2024, 1, 15), 1.5),
    _corp("ATLAS 5.500 06/15/31", "Atlas Bancorp", FIN, "A", 0.0550, D(2031, 6, 15), D(2026, 6, 15), 2.0),
    _corp("ATLAS 5.875 03/15/36", "Atlas Bancorp", FIN, "A", 0.05875, D(2036, 3, 15), D(2026, 3, 15), 1.75),
    _corp("MERID 5.800 04/15/30", "Meridian Financial", FIN, "BBB", 0.0580, D(2030, 4, 15), D(2025, 4, 15), 1.0),
    _corp("MERID 6.125 10/15/33", "Meridian Financial", FIN, "BBB", 0.06125, D(2033, 10, 15), D(2025, 10, 15), 1.25),
    _corp("KEYST 4.900 05/15/32", "Keystone Mutual Insurance", FIN, "AA", 0.0490, D(2032, 5, 15), D(2025, 5, 15), 1.0),
    _corp("KEYST 5.600 05/15/47", "Keystone Mutual Insurance", FIN, "AA", 0.0560, D(2047, 5, 15), D(2025, 5, 15), 1.0),
    _corp("VRTX 4.650 08/01/30", "Vertex Systems", TECH, "AA", 0.0465, D(2030, 8, 1), D(2025, 8, 1), 2.5),
    _corp("VRTX 5.100 05/01/35", "Vertex Systems", TECH, "AA", 0.0510, D(2035, 5, 1), D(2025, 5, 1), 2.0),
    _corp("VRTX 5.400 02/01/46", "Vertex Systems", TECH, "AA", 0.0540, D(2046, 2, 1), D(2026, 2, 1), 1.5),
    _corp("QNTM 5.350 12/01/32", "Quantum Devices", TECH, "A", 0.0535, D(2032, 12, 1), D(2025, 12, 1), 1.25),
    _corp("HALC 5.200 04/15/34", "Halcyon Health", HC, "A", 0.0520, D(2034, 4, 15), D(2024, 4, 15), 1.5),
    _corp("HALC 5.950 04/15/54", "Halcyon Health", HC, "A", 0.0595, D(2054, 4, 15), D(2024, 4, 15), 1.25),
    _corp("GRNT 5.900 11/15/29", "Granite Energy Partners", ENER, "BBB", 0.0590, D(2029, 11, 15), D(2024, 11, 15), 0.9),
    _corp("GRNT 6.250 07/15/35", "Granite Energy Partners", ENER, "BBB", 0.0625, D(2035, 7, 15), D(2025, 7, 15), 1.1),
    _corp("RIVR 5.450 01/15/36", "Riverstone Utilities", UTIL, "A", 0.0545, D(2036, 1, 15), D(2026, 1, 15), 0.8),
    _corp("RIVR 6.000 06/01/55", "Riverstone Utilities", UTIL, "A", 0.0600, D(2055, 6, 1), D(2025, 6, 1), 0.8),
    _corp("BLUW 6.050 03/01/33", "Bluewater Power", UTIL, "BBB", 0.0605, D(2033, 3, 1), D(2026, 3, 1), 0.75),
    _corp("IRON 5.750 09/15/31", "Ironclad Industries", IND, "BBB", 0.0575, D(2031, 9, 15), D(2026, 9, 15), 1.0),
    _corp("IRON 6.100 09/15/45", "Ironclad Industries", IND, "BBB", 0.0610, D(2045, 9, 15), D(2025, 9, 15), 0.9),
    _corp("SUMT 5.300 10/01/35", "Summit Rail", IND, "A", 0.0530, D(2035, 10, 1), D(2025, 10, 1), 1.0),
    _corp("HRVST 4.950 02/15/30", "Harvest Foods", STAP, "A", 0.0495, D(2030, 2, 15), D(2025, 2, 15), 1.2),
    _corp("HRVST 5.550 02/15/37", "Harvest Foods", STAP, "A", 0.0555, D(2037, 2, 15), D(2026, 2, 15), 1.0),
    _corp("SKYL 6.300 12/01/34", "Skyline Telecom", COMM, "BBB", 0.0630, D(2034, 12, 1), D(2024, 12, 1), 1.5),
    _corp("SKYL 6.650 12/01/54", "Skyline Telecom", COMM, "BBB", 0.0665, D(2054, 12, 1), D(2024, 12, 1), 1.25),
]
UNIVERSE: list[FixedRateBond] = TREASURIES + CORPORATES

# Simulated starting portfolio: market-value weights in percent of NAV. The
# book deliberately runs long duration (long-end Treasuries and 20-30Y credit)
# and breaks three policy limits (Atlas issuer > 6%, Financials > 18%,
# Treasuries < 35%) so the rebalance has real work to do.
INITIAL_WEIGHTS_PCT: dict[str, float] = {
    "UST 4.625 09/30/28": 4.0, "UST 5.000 09/30/31": 9.0, "UST 1.375 11/15/31": 4.0,
    "UST 5.250 08/15/36": 6.0, "UST STRIP 0 11/15/36": 2.0, "UST 4.750 02/15/45": 4.0,
    "UST 5.625 08/15/56": 4.0,
    "ATLAS 4.200 01/15/29": 1.0, "ATLAS 5.500 06/15/31": 3.5, "ATLAS 5.875 03/15/36": 3.0,
    "MERID 5.800 04/15/30": 2.0, "MERID 6.125 10/15/33": 4.0,
    "KEYST 4.900 05/15/32": 3.0, "KEYST 5.600 05/15/47": 3.0,
    "VRTX 4.650 08/01/30": 1.5, "VRTX 5.100 05/01/35": 1.0, "VRTX 5.400 02/01/46": 3.5,
    "QNTM 5.350 12/01/32": 3.0,
    "HALC 5.200 04/15/34": 2.5, "HALC 5.950 04/15/54": 3.5,
    "GRNT 5.900 11/15/29": 2.0, "GRNT 6.250 07/15/35": 3.5,
    "RIVR 5.450 01/15/36": 2.0, "RIVR 6.000 06/01/55": 3.5, "BLUW 6.050 03/01/33": 2.5,
    "IRON 5.750 09/15/31": 2.0, "IRON 6.100 09/15/45": 3.0, "SUMT 5.300 10/01/35": 3.0,
    "HRVST 4.950 02/15/30": 2.5, "HRVST 5.550 02/15/37": 3.0,
    "SKYL 6.300 12/01/34": 2.5, "SKYL 6.650 12/01/54": 3.0,
}

# Simulated benchmark: 60% Treasury / 40% IG corporate, weighted by simulated
# amount outstanding x price within each block.
BENCHMARK_BLOCKS = {TSY: 0.60, CORP: 0.40}

SPREAD_BASE_BP = {"AA": 55.0, "A": 85.0, "BBB": 125.0}
SECTOR_ADJ_BP = {FIN: 8.0, TECH: -6.0, HC: -2.0, ENER: 15.0, UTIL: 4.0, IND: 3.0, STAP: -4.0, COMM: 12.0}
IDIO_SD_BP = 6.0


@dataclass(frozen=True)
class Policy:
    """Simulated investment-policy limits for the rebalance."""

    max_issuer: float = 0.06  # corporate issuer, % of NAV
    max_sector: float = 0.18  # corporate sector, % of NAV
    min_treasury: float = 0.35
    max_turnover: float = 0.20  # one-way: sum of buys (= sum of sells) / NAV
    max_position: float = 0.10


def simulated_z_spreads(bonds: list[FixedRateBond], valuation: date, seed: int) -> np.ndarray:
    """SYNTHETIC z-spreads (decimal): rating base + term premium + sector + issuer noise."""
    rng = np.random.default_rng(seed)
    issuers = sorted({b.issuer for b in bonds if b.kind == CORP})
    idio = dict(zip(issuers, rng.normal(0.0, IDIO_SD_BP, len(issuers)), strict=True))
    out = []
    for b in bonds:
        if b.kind != CORP:
            out.append(0.0)
            continue
        t = (b.maturity - valuation).days / 365.25
        term = 2.5 * min(t, 10.0) + 0.8 * max(t - 10.0, 0.0)
        bp = SPREAD_BASE_BP[b.rating] + term + SECTOR_ADJ_BP[b.sector] + idio[b.issuer]
        out.append(bp * 1e-4)
    return np.array(out)


def transaction_cost_bp(b: FixedRateBond, valuation: date) -> float:
    """SYNTHETIC half bid-ask cost in bp of traded market value."""
    t = (b.maturity - valuation).days / 365.25
    if b.kind == TSY:
        return 0.5 + 0.1 * t
    return {"AA": 8.0, "A": 10.0, "BBB": 14.0}[b.rating] + 0.25 * t
