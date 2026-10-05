from datetime import date

import numpy as np
import pytest

from firisk.bootstrap import bootstrap_par_curve, par_instrument_prices, par_yield
from firisk.curve import ZeroCurve, flat_curve
from firisk.data import TENOR_YEARS, load_history


def test_curve_basics_and_forward_consistency():
    c = ZeroCurve(np.array([0.5, 2.0, 10.0, 30.0]), np.array([0.04, 0.042, 0.048, 0.05]))
    assert float(c.df(0.0)) == 1.0
    np.testing.assert_allclose(c.zero_rate(c.times), c.zeros, rtol=1e-14)
    t1, t2 = 3.0, 7.5
    f = float(c.forward_rate(t1, t2))
    assert float(c.df(t2)) == pytest.approx(float(c.df(t1)) * np.exp(-f * (t2 - t1)), rel=1e-14)
    simple = float(c.forward_rate(t1, t2, "simple"))
    assert float(c.df(t1) / c.df(t2)) == pytest.approx(1 + simple * (t2 - t1), rel=1e-14)
    assert float(c.df(40.0)) < float(c.df(30.0))  # flat-forward extrapolation keeps decaying


def test_flat_curve_and_compounding_conversion():
    c = flat_curve(0.05)
    np.testing.assert_allclose(c.zero_rate([0.25, 3.0, 25.0]), 0.05, rtol=1e-12)
    np.testing.assert_allclose(c.forward_rate(2.0, 9.0), 0.05, rtol=1e-12)
    assert float(c.zero_rate(1.0, "semi")) == pytest.approx(2 * (np.exp(0.025) - 1), rel=1e-14)
    with pytest.raises(ValueError):
        ZeroCurve(np.array([2.0, 1.0]), np.array([0.01, 0.02]))


def test_flat_par_curve_bootstraps_to_flat_semiannual_zeros():
    y = 0.05
    c = bootstrap_par_curve(TENOR_YEARS, np.full(len(TENOR_YEARS), y))
    coupon_tenors = TENOR_YEARS[TENOR_YEARS >= 0.5]
    np.testing.assert_allclose(c.zero_rate(coupon_tenors, "semi"), y, atol=1e-12)


def test_bootstrap_reprices_market_par_instruments():
    par = load_history().par_yields_on(date(2026, 10, 1))
    c = bootstrap_par_curve(TENOR_YEARS, par)
    assert np.abs(par_instrument_prices(c, TENOR_YEARS, par) - 100).max() < 1e-10
    for t, y in zip(TENOR_YEARS[3:], par[3:], strict=True):
        assert par_yield(c, t) == pytest.approx(y, abs=1e-12)
    grid = np.linspace(0.01, 30, 400)
    assert np.all(np.diff(c.df(grid)) < 0)
