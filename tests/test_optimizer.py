from dataclasses import replace

import numpy as np
import pytest

from firisk.engine import execute_trades
from firisk.optimizer import InfeasibleRebalance, check_constraints, objective_terms, solve_rebalance
from firisk.universe import UNIVERSE, Policy


def test_every_constraint_holds_after_rounding_and_costs(analysis):
    c = analysis.constraints_after
    assert c["pass"].all(), c
    pol = analysis.rebalance_terms["problem"].policy
    r = analysis.rebalanced
    assert r.group_weights("issuer").max() <= pol.max_issuer
    assert r.group_weights("sector").max() <= pol.max_sector
    assert r.treasury_weight >= pol.min_treasury
    assert analysis.rebalance_terms["turnover"] <= pol.max_turnover
    assert r.weights.min() >= 0 and r.weights.max() <= pol.max_position
    assert (r.face >= 0).all()


def test_starting_book_breaches_and_rebalance_improves_target_fit(analysis):
    assert (~analysis.constraints_before["pass"]).sum() == 3
    t = analysis.rebalance_terms
    assert t["after"]["krd_l1_gap"] < 0.25 * t["before"]["krd_l1_gap"]
    assert abs(t["after"]["duration_gap"]) < abs(t["before"]["duration_gap"])


def test_trades_are_self_financing(analysis):
    tr = analysis.trades
    r = analysis.rebalanced
    assert 0 <= r.cash < 5_000
    assert r.nav == pytest.approx(analysis.portfolio.nav - tr["cost_usd"].sum(), abs=1e-6)
    assert (tr["trade_face"] % 1000 == 0).all()


def test_lp_solution_satisfies_constraints_before_rounding(analysis):
    prob = analysis.rebalance_terms["problem"]
    dw = solve_rebalance(prob)
    w = prob.w0 + dw
    turnover = dw[dw > 0].sum()
    assert check_constraints(w, turnover, prob, tol=1e-9)["pass"].all()
    assert np.abs(dw).sum() > 0


def test_costs_trade_off_against_tracking(analysis):
    prob = analysis.rebalance_terms["problem"]
    no_cost = replace(prob, include_costs=False)
    w_cost = prob.w0 + solve_rebalance(prob)
    w_free = prob.w0 + solve_rebalance(no_cost)
    # Without costs the optimizer can only track the target at least as well.
    assert objective_terms(w_free, no_cost)["krd_l1_gap"] <= objective_terms(w_cost, prob)["krd_l1_gap"] + 1e-9
    cost_free = (np.abs(w_free - prob.w0) * prob.cost).sum()
    cost_paid = (np.abs(w_cost - prob.w0) * prob.cost).sum()
    assert cost_paid <= cost_free + 1e-12


def test_tighter_turnover_limit_is_respected(analysis):
    prob = replace(analysis.rebalance_terms["problem"],
                   policy=Policy(max_turnover=0.08, min_treasury=0.35, max_issuer=0.06, max_sector=0.18))
    dw = solve_rebalance(prob)
    assert dw[dw > 0].sum() <= 0.08
    reb, trades = execute_trades(analysis.portfolio, dw, prob.cost, analysis.risk.frame["dirty"].to_numpy(), UNIVERSE)
    turnover = trades.loc[trades["trade_mv"] > 0, "trade_mv"].sum() / analysis.portfolio.nav
    assert check_constraints(reb.weights, turnover, prob)["pass"].all()


def test_infeasible_policy_raises(analysis):
    prob = replace(analysis.rebalance_terms["problem"], policy=Policy(min_treasury=0.9, max_turnover=0.05))
    with pytest.raises(InfeasibleRebalance):
        solve_rebalance(prob)
