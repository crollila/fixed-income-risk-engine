import numpy as np
import pytest

from firisk.bond import CORP, TSY
from firisk.data import TENOR_LABELS
from firisk.market import BondBook
from firisk.universe import BENCHMARK_BLOCKS, SIMULATED_NAV, UNIVERSE


def test_key_rate_dv01s_reconcile_to_parallel_dv01_per_bond(analysis):
    f = analysis.risk.frame
    rel = np.abs(analysis.risk.kr_dv01.sum(axis=1) / f["dv01"].to_numpy() - 1)
    assert rel.max() < 1e-4


def test_key_rate_bucket_far_from_maturity_has_no_risk(analysis):
    f = analysis.risk.frame.set_index("bond_id")
    i = list(f.index).index("UST 4.625 09/30/28")
    assert abs(analysis.risk.kr_dv01[i, TENOR_LABELS.index("30Y")]) < 1e-12
    assert abs(analysis.risk.kr_dv01[i, TENOR_LABELS.index("10Y")]) < 1e-12


def test_effective_duration_close_to_modified_for_bullets(analysis):
    f = analysis.risk.frame
    assert np.allclose(f["eff_duration"], f["modified"], rtol=0.025)


def test_cs01_zero_for_treasuries_and_close_to_dv01_for_corporates(analysis):
    f = analysis.risk.frame
    tsy, corp = f["kind"] == TSY, f["kind"] == CORP
    assert (f.loc[tsy, "cs01"] == 0).all()
    assert (f.loc[corp, "cs01"] > 0).all()
    # z-spread and zero-rate shifts discount identically, so spread duration ~ effective duration
    assert np.allclose(f.loc[corp, "spread_duration"], f.loc[corp, "eff_duration"], rtol=0.05)


def test_z_spreads_recover_simulated_spreads(analysis):
    book = BondBook(UNIVERSE, analysis.market.valuation_date)
    z = book.z_spreads(analysis.market.curve, analysis.risk.frame["dirty"].to_numpy())
    np.testing.assert_allclose(z, analysis.z, atol=1e-12)


def test_portfolio_risk_equals_sum_of_constituents(analysis):
    p = analysis.portfolio
    pos = p.positions
    v = analysis.validation
    assert v["portfolio_dv01_full_reprice"] == pytest.approx(pos["dv01"].sum(), rel=1e-10)
    assert p.cs01 == pytest.approx(pos["cs01"].sum(), rel=1e-12)
    kr_cols = [f"krdv01_{t}" for t in TENOR_LABELS]
    np.testing.assert_allclose(pos[kr_cols].sum().to_numpy(), p.kr_dv01, rtol=1e-12)
    assert p.nav == pytest.approx(pos["market_value"].sum(), rel=1e-12)
    assert p.kr_dv01.sum() == pytest.approx(p.dv01, rel=1e-4)


def test_simulated_book_size_weights_and_groups(analysis):
    p = analysis.portfolio
    assert abs(p.nav / SIMULATED_NAV - 1) < 1e-4
    assert p.weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert p.group_weights("sector").sum() + p.treasury_weight == pytest.approx(1.0, abs=1e-12)
    assert p.group_weights("issuer").sum() == pytest.approx(p.group_weights("rating").sum(), abs=1e-12)
    lad = p.ladder()
    assert (lad["treasury_mv"] + lad["corporate_mv"]).sum() == pytest.approx(p.nav, rel=1e-12)


def test_benchmark_blocks(analysis):
    b = analysis.benchmark
    assert b.treasury_weight == pytest.approx(BENCHMARK_BLOCKS[TSY], abs=1e-9)
    assert b.nav == pytest.approx(analysis.portfolio.nav, rel=1e-12)
