"""PCA of daily Treasury par-curve changes, mapped onto portfolio key-rate risk.

Because key-rate DV01s are measured against the same par tenors the PCA is
run on, factor exposures are just dot products: the P&L of a one-standard-
deviation move in PC j is ``-KRDV01 . (sqrt(lambda_j) e_j)`` (in bp).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

Z99 = 2.3263478740408408  # standard normal 99% quantile


@dataclass
class CurvePCA:
    tenor_labels: list[str]
    tenor_years: np.ndarray
    loadings: np.ndarray  # (n_tenors, n_tenors), column j = PC j (unit vectors)
    eigenvalues: np.ndarray  # variance of daily changes, bp^2
    mean_change_bp: np.ndarray
    cov_bp2: np.ndarray
    n_obs: int
    start: str
    end: str

    @property
    def explained(self) -> np.ndarray:
        return self.eigenvalues / self.eigenvalues.sum()

    def shape_label(self, j: int) -> str:
        return classify_shape(self.loadings[:, j])

    def factor_scenario_bp(self, j: int, n_sigma: float, horizon_days: int = 1) -> np.ndarray:
        return n_sigma * np.sqrt(self.eigenvalues[j] * horizon_days) * self.loadings[:, j]


def daily_changes_bp(levels_pct: pd.DataFrame) -> pd.DataFrame:
    return levels_pct.diff().dropna(how="any") * 100.0


def fit_pca(changes_bp: pd.DataFrame, tenor_years: np.ndarray) -> CurvePCA:
    x = changes_bp.to_numpy(dtype=float)
    mean = x.mean(axis=0)
    xc = x - mean
    cov = xc.T @ xc / (len(x) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    vecs = _orient(vecs, np.asarray(tenor_years))
    return CurvePCA(list(changes_bp.columns), np.asarray(tenor_years), vecs, vals, mean, cov,
                    len(x), str(changes_bp.index[0].date()), str(changes_bp.index[-1].date()))


def _orient(vecs: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Sign conventions (eigenvectors are sign-free):
    PC1 positive on average (rates up), PC2 long end above short end
    (steepening), PC3 belly (3-10Y) above the wings (belly cheapens)."""
    v = vecs.copy()
    if v[:, 0].sum() < 0:
        v[:, 0] *= -1
    if v[-1, 1] - v[0, 1] < 0:
        v[:, 1] *= -1
    belly = (t >= 3) & (t <= 10)
    if v[belly, 2].mean() - v[~belly, 2].mean() < 0:
        v[:, 2] *= -1
    return v


def classify_shape(loading: np.ndarray, rel_tol: float = 0.15) -> str:
    """Empirical label from sign changes across tenors (small loadings ignored)."""
    big = loading[np.abs(loading) >= rel_tol * np.abs(loading).max()]
    changes = int(np.sum(np.diff(np.sign(big)) != 0))
    return {0: "level", 1: "slope", 2: "curvature"}.get(changes, f"{changes} sign changes")


def factor_risk(pca: CurvePCA, kr_dv01: np.ndarray, n_factors: int = 3) -> dict:
    """Daily rate-risk decomposition of a KR-DV01 vector ($ per bp at each par tenor)."""
    expo = kr_dv01 @ pca.loadings  # $ per 1bp move along each PC
    sigma_pc = np.sqrt(pca.eigenvalues)
    one_sigma_pnl = -expo * sigma_pc
    var_by_pc = (expo * sigma_pc) ** 2
    total_var = float(kr_dv01 @ pca.cov_bp2 @ kr_dv01)
    return {
        "exposure_per_bp": expo[:n_factors],
        "one_sigma_daily_pnl": one_sigma_pnl[:n_factors],
        "variance_share": var_by_pc[:n_factors] / total_var,
        "daily_sigma_total": np.sqrt(total_var),
        "daily_sigma_3pc": float(np.sqrt(var_by_pc[:n_factors].sum())),
        "var99_parametric": Z99 * np.sqrt(total_var),
    }


def historical_var(changes_bp: pd.DataFrame, kr_dv01: np.ndarray, q: float = 0.01) -> dict:
    """First-order historical-simulation P&L: each past daily curve change applied to today's KR-DV01."""
    pnl = -(changes_bp.to_numpy() @ kr_dv01)
    return {"var99_historical": float(-np.quantile(pnl, q)), "worst_day": float(pnl.min()),
            "worst_day_date": str(changes_bp.index[int(np.argmin(pnl))].date())}
