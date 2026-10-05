"""End-to-end analysis: market -> instruments -> portfolio risk -> stress -> PCA -> rebalance."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from firisk.bond import TSY
from firisk.bootstrap import par_instrument_prices
from firisk.config import SEED, VALUATION_DATE
from firisk.data import TENOR_LABELS, TENOR_YEARS, CurveHistory, load_history
from firisk.market import BondBook, Market
from firisk.optimizer import RebalanceProblem, check_constraints, objective_terms, solve_rebalance
from firisk.pca import CurvePCA, daily_changes_bp, factor_risk, fit_pca, historical_var
from firisk.portfolio import Portfolio, benchmark_weights, faces_from_weights
from firisk.risk import RiskTable, bond_risk
from firisk.scenarios import Scenario, StressEngine, standard_scenarios
from firisk.universe import (
    BENCHMARK_BLOCKS,
    FACE_ROUNDING,
    INITIAL_WEIGHTS_PCT,
    SIMULATED_NAV,
    UNIVERSE,
    Policy,
    simulated_z_spreads,
    transaction_cost_bp,
)

# PCA runs on the coupon curve (1Y-30Y). Bill tenors carry idiosyncratic
# money-market noise (debt-ceiling episodes, policy expectations) that would
# otherwise take a factor of its own; ANALYSIS.md reports the 11-tenor check.
PCA_SLICE = slice(3, None)
PCA_TENORS = TENOR_LABELS[PCA_SLICE]
CONVERGENCE_SHOCKS_BP = [-200, -100, -50, -25, -10, -5, -2, -1, 1, 2, 5, 10, 25, 50, 100, 200]


@dataclass
class Analysis:
    history: CurveHistory
    market: Market
    z: np.ndarray
    risk: RiskTable
    portfolio: Portfolio
    benchmark: Portfolio
    rebalanced: Portfolio
    stress: pd.DataFrame
    stress_after: pd.DataFrame
    convergence: pd.DataFrame
    pca: CurvePCA
    pca_all_tenors: CurvePCA
    changes_bp: pd.DataFrame
    pca_scenarios: pd.DataFrame
    factor_before: dict
    factor_after: dict
    hist_var_before: dict
    hist_var_after: dict
    trades: pd.DataFrame
    constraints_before: pd.DataFrame
    constraints_after: pd.DataFrame
    rebalance_terms: dict
    validation: dict


def _pad_bills(shift_coupon_curve: np.ndarray) -> np.ndarray:
    """Extend a 1Y-30Y factor move to the bill tenors, flat from the 1Y point."""
    n_bill = len(TENOR_LABELS) - len(shift_coupon_curve)
    return np.concatenate([np.full(n_bill, shift_coupon_curve[0]), shift_coupon_curve])


def build_market(history: CurveHistory, when: date = VALUATION_DATE) -> Market:
    return Market(when, TENOR_YEARS, history.par_yields_on(when))


def run_analysis(history: CurveHistory | None = None, policy: Policy | None = None,
                 include_costs: bool = True) -> Analysis:
    history = history or load_history()
    policy = policy or Policy()
    market = build_market(history)
    bonds = UNIVERSE
    z = simulated_z_spreads(bonds, market.valuation_date, SEED)
    risk = bond_risk(bonds, market, z, TENOR_LABELS)
    f = risk.frame
    dirty = f["dirty"].to_numpy()

    w_init = np.array([INITIAL_WEIGHTS_PCT.get(b.bond_id, 0.0) / 100.0 for b in bonds])
    port = Portfolio("Simulated portfolio", faces_from_weights(w_init, dirty, SIMULATED_NAV, FACE_ROUNDING), risk)
    outstanding = np.array([b.outstanding_bn for b in bonds])
    w_bench = benchmark_weights(f, outstanding, BENCHMARK_BLOCKS)
    bench = Portfolio("Simulated benchmark", faces_from_weights(w_bench, dirty, port.nav, 0.0), risk)

    # ----- stress -----------------------------------------------------------------
    engine = StressEngine(bonds, market, z, risk)
    scenarios = standard_scenarios(market.tenors)
    stress = engine.run_all(port.face, scenarios, port.nav)
    convergence = engine.convergence(port.face, CONVERGENCE_SHOCKS_BP)

    # ----- PCA --------------------------------------------------------------------
    changes = daily_changes_bp(history.frame[TENOR_LABELS])
    changes = changes[changes.index <= pd.Timestamp(market.valuation_date)]
    pca = fit_pca(changes[PCA_TENORS], TENOR_YEARS[PCA_SLICE])
    pca_all = fit_pca(changes, TENOR_YEARS)
    pca_sc = [Scenario(f"PC{j + 1} {pca.shape_label(j)} {s:+d} sigma (1M)",
                       _pad_bills(pca.factor_scenario_bp(j, s, 21)), family="pca")
              for j in range(3) for s in (2, -2)]

    # ----- rebalance --------------------------------------------------------------
    krd = risk.kr_dv01 / (dirty * 1e-4)[:, None]
    is_tsy = (f["kind"] == TSY).to_numpy()
    prob = RebalanceProblem(
        w0=port.weights, duration=f["eff_duration"].to_numpy(), krd=krd,
        target_krd=bench.weights @ krd, target_duration=bench.duration,
        issuer=np.where(is_tsy, "", f["issuer"].to_numpy()),
        sector=np.where(is_tsy, "", f["sector"].to_numpy()), is_treasury=is_tsy,
        cost=np.array([transaction_cost_bp(b, market.valuation_date) for b in bonds]) * 1e-4,
        policy=policy, include_costs=include_costs,
    )
    dw = solve_rebalance(prob)
    reb, trades = execute_trades(port, dw, prob.cost, dirty, bonds)
    turnover = float(trades.loc[trades["trade_mv"] > 0, "trade_mv"].sum() / port.nav)
    cons_after = check_constraints(reb.weights, turnover, prob)
    cons_before = check_constraints(port.weights, 0.0, prob)
    terms = {"before": objective_terms(port.weights, prob), "after": objective_terms(reb.weights, prob),
             "turnover": turnover, "cost_usd": float(trades["cost_usd"].sum()),
             "n_trades": int((trades["trade_face"] != 0).sum()), "problem": prob}
    stress_after = engine.run_all(reb.face, scenarios, reb.nav)
    pca_frame = pd.concat([engine.run_all(port.face, pca_sc, port.nav).assign(book="before"),
                           engine.run_all(reb.face, pca_sc, reb.nav).assign(book="after")],
                          ignore_index=True)

    # ----- self-validation ----------------------------------------------------------
    book = BondBook(bonds, market.valuation_date)
    held = port.face / 100.0

    def port_value(mkt: Market) -> float:
        return float(held @ book.dirty(mkt.curve, z))

    port_dv01_full = -(port_value(market.shifted(1.0)) - port_value(market.shifted(-1.0))) / 2.0
    reprice = par_instrument_prices(market.curve, market.tenors, market.par_yields)
    validation = {
        "bootstrap_max_abs_reprice_error": float(np.abs(reprice - 100).max()),
        "portfolio_dv01_full_reprice": port_dv01_full,
        "portfolio_dv01_sum_of_positions": port.dv01,
        "kr_sum_vs_parallel_rel_diff": float(port.kr_dv01.sum() / port.dv01 - 1.0),
        "max_bond_kr_sum_vs_parallel_rel_diff": float(np.abs(risk.kr_dv01.sum(axis=1) / f["dv01"] - 1).max()),
        "all_constraints_pass_after": bool(cons_after["pass"].all()),
    }
    return Analysis(history, market, z, risk, port, bench, reb, stress, stress_after, convergence,
                    pca, pca_all, changes, pca_frame,
                    factor_risk(pca, port.kr_dv01[PCA_SLICE]), factor_risk(pca, reb.kr_dv01[PCA_SLICE]),
                    historical_var(changes, port.kr_dv01), historical_var(changes, reb.kr_dv01),
                    trades, cons_before, cons_after, terms, validation)


def execute_trades(port: Portfolio, dw: np.ndarray, cost_rate: np.ndarray, dirty: np.ndarray,
                   bonds) -> tuple[Portfolio, pd.DataFrame]:
    """Turn weight trades into $1,000-rounded face trades; costs and rounding go to cash."""
    nav = port.nav
    dface = np.round(dw * nav / (dirty / 100.0) / FACE_ROUNDING) * FACE_ROUNDING
    dface[np.abs(dw) < 1e-7] = 0.0
    new_face = np.maximum(port.face + dface, 0.0)
    dface = new_face - port.face
    trade_mv = dface * dirty / 100.0
    cost = np.abs(trade_mv) * cost_rate
    cash = port.cash - trade_mv.sum() - cost.sum()
    while cash < 0:  # trim the largest buy so the trade list is self-financing
        i = int(np.argmax(trade_mv))
        dface[i] -= FACE_ROUNDING
        new_face[i] -= FACE_ROUNDING
        trade_mv[i] = dface[i] * dirty[i] / 100.0
        cost[i] = abs(trade_mv[i]) * cost_rate[i]
        cash = port.cash - trade_mv.sum() - cost.sum()
    reb = Portfolio("Rebalanced portfolio", new_face, port.risk, cash=cash)
    side = np.where(dface > 0, "BUY", np.where(dface < 0, "SELL", ""))
    trades = pd.DataFrame({"bond_id": [b.bond_id for b in bonds], "side": side, "trade_face": dface,
                           "trade_mv": trade_mv, "cost_usd": cost,
                           "weight_before": port.weights, "weight_after": reb.weights})
    return reb, trades
