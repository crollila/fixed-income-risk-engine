import numpy as np
import pytest

from firisk.data import TENOR_YEARS
from firisk.scenarios import CREDIT_WIDENING, Scenario, StressEngine, parallel, twist
from firisk.universe import UNIVERSE


def test_approximation_errors_converge_at_second_and_third_order(analysis):
    c = analysis.convergence
    for sign in (1, -1):
        s = c[(c["shock_bp"] * sign >= 2) & (c["shock_bp"] * sign <= 50)]
        x = np.log(s["shock_bp"].abs())
        slope1 = np.polyfit(x, np.log(s["abs_err_first"]), 1)[0]
        slope2 = np.polyfit(x, np.log(s["abs_err_second"]), 1)[0]
        assert 1.9 < slope1 < 2.1  # duration-only error is O(dy^2)
        assert 2.7 < slope2 < 3.3  # convexity-corrected error is O(dy^3)
    one = c.set_index("shock_bp").loc[1]
    assert one["abs_err_first"] / abs(one["full_pnl"]) < 1e-3


def test_second_order_beats_first_order_in_every_scenario(analysis):
    s = analysis.stress
    assert (s["second_order_error"].abs() < s["first_order_error"].abs()).all()
    # positive convexity: the linear estimate is always too pessimistic
    assert (s["first_order_error"] < 0).all()


def test_attribution_components_sum_to_total(analysis):
    s = analysis.stress
    np.testing.assert_allclose(s["rates_pnl"] + s["spread_pnl"] + s["interaction_pnl"], s["full_pnl"], atol=1e-6)
    pure_rates = s[s["family"].isin(["rates", "curve"])]
    assert (pure_rates["spread_pnl"].abs() < 1e-6).all()


def test_convexity_asymmetry_of_parallel_shocks(analysis):
    s = analysis.stress.set_index("scenario")
    for bp in (25, 50, 100):
        assert s.loc[f"Parallel -{bp}bp", "full_pnl"] > -s.loc[f"Parallel +{bp}bp", "full_pnl"] > 0


def test_twist_shape():
    tw = twist(-25, 25, TENOR_YEARS)
    assert tw[TENOR_YEARS <= 2].tolist() == [-25.0] * int((TENOR_YEARS <= 2).sum())
    assert tw[TENOR_YEARS >= 10].tolist() == [25.0] * int((TENOR_YEARS >= 10).sum())
    assert tw[list(TENOR_YEARS).index(5.0)] == pytest.approx(-25 + 50 * 3 / 8)


def test_spread_shock_leaves_treasuries_unchanged(analysis):
    eng = StressEngine(UNIVERSE, analysis.market, analysis.z, analysis.risk)
    face = np.full(len(UNIVERSE), 1_000_000.0)
    ds = np.full(len(UNIVERSE), 50.0)
    pnl = eng.full_pnl(face, parallel(0, TENOR_YEARS), ds)
    is_tsy = (analysis.risk.frame["kind"] == "TSY").to_numpy()
    assert np.all(pnl[is_tsy] < 0)  # raw z-bump also hits Treasuries ...
    sc_ds = eng.spread_vector_bp(Scenario("credit", parallel(0, TENOR_YEARS), CREDIT_WIDENING))
    assert np.all(sc_ds[is_tsy] == 0)  # ... but scenarios only shock credit
