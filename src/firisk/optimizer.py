"""Constrained rebalance toward a target duration / key-rate-duration profile.

Linear program (HiGHS via ``scipy.optimize.linprog``) in market-value weights.
Decision variables are buys ``b >= 0`` and sells ``s >= 0`` so the new weights
are ``w = w0 + b - s``. Risk is linear in weights (portfolio KRD_k = sum_i w_i
KRD_ik), so tracking error to the target profile is an L1 norm with slacks:

    minimise  sum_k |KRD_k(w) - T_k| + lambda_D |D(w) - D_T|
              + lambda_c * sum_i c_i (b_i + s_i)          (optional costs)
              + eps * sum_i (b_i + s_i)                     (no needless trades)

    s.t.  self-financing: sum_i b_i (1 + c_i) = sum_i s_i (1 - c_i)
          long only, per-position cap, corporate issuer cap, corporate sector cap,
          minimum Treasury weight, one-way turnover sum_i b_i <= limit.

Hard limits are tightened by a small ``buffer`` so that rounding trades to
$1,000 face and paying costs cannot push a binding constraint over its limit;
:func:`check_constraints` verifies the final rounded book independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from firisk.universe import Policy


class InfeasibleRebalance(RuntimeError):
    pass


@dataclass
class RebalanceProblem:
    w0: np.ndarray
    duration: np.ndarray  # effective duration per bond
    krd: np.ndarray  # (n, K) key-rate durations per bond
    target_krd: np.ndarray  # (K,)
    target_duration: float
    issuer: np.ndarray  # issuer name per bond ("" for Treasuries)
    sector: np.ndarray  # sector per bond ("" for Treasuries)
    is_treasury: np.ndarray
    cost: np.ndarray  # decimal cost per unit MV traded
    policy: Policy
    include_costs: bool = True
    cost_aversion: float = 100.0
    duration_weight: float = 1.0
    eps: float = 1e-3
    buffer: float = 1e-4


def solve_rebalance(p: RebalanceProblem) -> np.ndarray:
    """Return the trade vector ``dw = b - s`` in NAV weights."""
    n, K = p.krd.shape
    c = p.cost if p.include_costs else np.zeros(n)
    nv = 2 * n + 2 * K + 2
    ib, is_, iup, idn, idp, idm = 0, n, 2 * n, 2 * n + K, 2 * n + 2 * K, 2 * n + 2 * K + 1

    obj = np.zeros(nv)
    trade_pen = p.eps + (p.cost_aversion * c if p.include_costs else 0.0)
    obj[ib:ib + n] = trade_pen
    obj[is_:is_ + n] = trade_pen
    obj[iup:iup + K] = 1.0
    obj[idn:idn + K] = 1.0
    obj[idp] = obj[idm] = p.duration_weight

    A_eq, b_eq = [], []
    row = np.zeros(nv)
    row[ib:ib + n] = 1.0 + c
    row[is_:is_ + n] = -(1.0 - c)
    A_eq.append(row)
    b_eq.append(0.0)
    krd0 = p.w0 @ p.krd
    for k in range(K):
        row = np.zeros(nv)
        row[ib:ib + n] = p.krd[:, k]
        row[is_:is_ + n] = -p.krd[:, k]
        row[iup + k], row[idn + k] = -1.0, 1.0
        A_eq.append(row)
        b_eq.append(p.target_krd[k] - krd0[k])
    row = np.zeros(nv)
    row[ib:ib + n] = p.duration
    row[is_:is_ + n] = -p.duration
    row[idp], row[idm] = -1.0, 1.0
    A_eq.append(row)
    b_eq.append(p.target_duration - p.w0 @ p.duration)

    A_ub, b_ub = [], []

    def add_group(mask: np.ndarray, limit: float, sign: float = 1.0) -> None:
        r = np.zeros(nv)
        r[ib:ib + n] = sign * mask
        r[is_:is_ + n] = -sign * mask
        A_ub.append(r)
        b_ub.append(sign * (limit - p.w0[mask].sum()) if sign > 0 else p.w0[mask].sum() - limit)

    for i in range(n):  # long only and position cap
        one = np.zeros(n, dtype=bool)
        one[i] = True
        add_group(one, p.policy.max_position - p.buffer)
        add_group(one, 0.0, sign=-1.0)
    corp = ~p.is_treasury
    for name in sorted(set(p.issuer[corp])):
        add_group(corp & (p.issuer == name), p.policy.max_issuer - p.buffer)
    for name in sorted(set(p.sector[corp])):
        add_group(corp & (p.sector == name), p.policy.max_sector - p.buffer)
    add_group(p.is_treasury, p.policy.min_treasury + p.buffer, sign=-1.0)
    r = np.zeros(nv)
    r[ib:ib + n] = 1.0
    A_ub.append(r)
    b_ub.append(p.policy.max_turnover - p.buffer)

    res = linprog(obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=np.array(A_eq),
                  b_eq=np.array(b_eq), bounds=[(0, None)] * nv, method="highs")
    if res.status != 0:
        raise InfeasibleRebalance(res.message)
    x = res.x
    return x[ib:ib + n] - x[is_:is_ + n]


def objective_terms(w: np.ndarray, p: RebalanceProblem) -> dict[str, float]:
    return {
        "krd_l1_gap": float(np.abs(w @ p.krd - p.target_krd).sum()),
        "duration_gap": float(w @ p.duration - p.target_duration),
    }


def check_constraints(w: np.ndarray, turnover: float, p: RebalanceProblem,
                      tol: float = 1e-9) -> pd.DataFrame:
    """Independent post-trade check of every policy limit on final weights."""
    corp = ~p.is_treasury
    rows = []

    def add(name: str, value: float, limit: float, kind: str) -> None:
        ok = value <= limit + tol if kind == "max" else value >= limit - tol
        rows.append({"constraint": name, "value": value, "limit": limit, "type": kind, "pass": bool(ok)})

    issuers = pd.Series(w[corp]).groupby(p.issuer[corp]).sum()
    sectors = pd.Series(w[corp]).groupby(p.sector[corp]).sum()
    add(f"Max issuer weight ({issuers.idxmax()})", float(issuers.max()), p.policy.max_issuer, "max")
    add(f"Max sector weight ({sectors.idxmax()})", float(sectors.max()), p.policy.max_sector, "max")
    add("Min Treasury weight", float(w[p.is_treasury].sum()), p.policy.min_treasury, "min")
    add("One-way turnover", turnover, p.policy.max_turnover, "max")
    add("Max single position", float(w.max()), p.policy.max_position, "max")
    add("Long only (min weight)", float(w.min()), 0.0, "min")
    return pd.DataFrame(rows)
