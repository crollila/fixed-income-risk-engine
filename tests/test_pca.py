import numpy as np
import pandas as pd
import pytest

from firisk.pca import classify_shape, factor_risk, fit_pca

T = np.array([1, 2, 3, 5, 7, 10, 20, 30], dtype=float)


def synthetic_changes(n=20000, seed=11):
    """Independent fixture: data generated from KNOWN orthonormal factors and variances."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(len(T), len(T))))
    lam = np.array([100.0, 25.0, 6.0, 1.0, 0.5, 0.25, 0.1, 0.05])
    x = rng.normal(size=(n, len(T))) * np.sqrt(lam) @ q.T + 3.0  # nonzero mean must be removed
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame(x, index=idx, columns=[f"{int(t)}Y" for t in T]), q, lam


def test_pca_recovers_known_factors():
    df, q, lam = synthetic_changes()
    p = fit_pca(df, T)
    for j in range(3):
        assert abs(p.loadings[:, j] @ q[:, j]) > 0.995
    np.testing.assert_allclose(p.explained[:3], lam[:3] / lam.sum(), rtol=0.05)
    np.testing.assert_allclose(p.mean_change_bp, 3.0, atol=0.1)


def test_eigen_structure_is_consistent():
    df, _, _ = synthetic_changes(n=3000)
    p = fit_pca(df, T)
    assert p.explained.sum() == pytest.approx(1.0)
    assert np.all(np.diff(p.eigenvalues) <= 0)
    np.testing.assert_allclose(p.loadings.T @ p.loadings, np.eye(len(T)), atol=1e-12)
    np.testing.assert_allclose(p.loadings @ np.diag(p.eigenvalues) @ p.loadings.T, p.cov_bp2, atol=1e-9)


def test_shape_classification():
    assert classify_shape(np.array([0.3, 0.35, 0.4, 0.4, 0.38, 0.33])) == "level"
    assert classify_shape(np.array([-0.5, -0.3, -0.1, 0.1, 0.3, 0.5])) == "slope"
    assert classify_shape(np.array([-0.5, 0.2, 0.5, 0.3, -0.2, -0.4])) == "curvature"


def test_real_curve_factors_are_level_slope_curvature(analysis):
    p = analysis.pca
    assert [p.shape_label(j) for j in range(3)] == ["level", "slope", "curvature"]
    assert (p.loadings[:, 0] > 0).all()
    assert p.loadings[-1, 1] > 0 > p.loadings[0, 1]
    assert p.explained[0] > 0.7 and p.explained[:3].sum() > 0.95


def test_factor_risk_decomposition_is_exact(analysis):
    p = analysis.pca
    g = analysis.portfolio.kr_dv01[3:]
    fr = factor_risk(p, g, n_factors=len(g))
    assert fr["variance_share"].sum() == pytest.approx(1.0, abs=1e-10)
    assert fr["daily_sigma_total"] == pytest.approx(np.sqrt(g @ p.cov_bp2 @ g))
    shock = p.loadings[:, 0] * np.sqrt(p.eigenvalues[0])
    assert fr["one_sigma_daily_pnl"][0] == pytest.approx(-(g @ shock))
