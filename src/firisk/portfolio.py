"""Portfolio construction and aggregation (position risk = face/100 x bond risk)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from firisk.bond import CORP, TSY
from firisk.risk import RiskTable

LADDER_EDGES = [0, 1, 3, 5, 7, 10, 20, 31]
LADDER_LABELS = ["0-1y", "1-3y", "3-5y", "5-7y", "7-10y", "10-20y", "20-30y"]


@dataclass
class Portfolio:
    name: str
    face: np.ndarray  # face amount per universe bond ($); zeros allowed
    risk: RiskTable  # universe bond risk per 100 face
    cash: float = 0.0

    @property
    def held(self) -> np.ndarray:
        return self.face != 0

    @property
    def positions(self) -> pd.DataFrame:
        f = self.risk.frame
        scale = self.face / 100.0
        df = f[["bond_id", "issuer", "sector", "rating", "kind", "coupon", "maturity",
                "years_to_maturity", "clean", "dirty", "ytm", "z_spread_bp",
                "eff_duration", "eff_convexity", "spread_duration"]].copy()
        df.insert(5, "face", self.face)
        df["market_value"] = scale * f["dirty"].to_numpy()
        df["dv01"] = scale * f["dv01"].to_numpy()
        df["cs01"] = scale * f["cs01"].to_numpy()
        df["weight"] = df["market_value"] / self.nav
        for k, lab in enumerate(self.risk.tenor_labels):
            df[f"krdv01_{lab}"] = scale * self.risk.kr_dv01[:, k]
        return df[self.held].reset_index(drop=True)

    # ----- headline aggregates ---------------------------------------------------
    @property
    def market_values(self) -> np.ndarray:
        return self.face / 100.0 * self.risk.frame["dirty"].to_numpy()

    @property
    def nav(self) -> float:
        return float(self.market_values.sum() + self.cash)

    @property
    def weights(self) -> np.ndarray:
        return self.market_values / self.nav

    def _sum(self, col: str) -> float:
        return float((self.face / 100.0 * self.risk.frame[col].to_numpy()).sum())

    @property
    def dv01(self) -> float:
        return self._sum("dv01")

    @property
    def cs01(self) -> float:
        return self._sum("cs01")

    @property
    def kr_dv01(self) -> np.ndarray:
        return (self.face / 100.0) @ self.risk.kr_dv01

    @property
    def duration(self) -> float:
        """Effective duration of NAV (cash has zero duration)."""
        return self.dv01 / (self.nav * 1e-4)

    @property
    def spread_duration(self) -> float:
        return self.cs01 / (self.nav * 1e-4)

    @property
    def convexity(self) -> float:
        return float((self.weights * self.risk.frame["eff_convexity"].to_numpy()).sum())

    @property
    def ytm(self) -> float:
        """MV-weighted yield (a standard approximation of portfolio yield)."""
        w = self.weights
        return float((w * self.risk.frame["ytm"].to_numpy()).sum() / w.sum())

    def dts(self) -> float:
        """Duration-times-spread (years x bp) of the credit book, NAV-weighted."""
        f = self.risk.frame
        return float((self.weights * f["spread_duration"].to_numpy() * f["z_spread_bp"].to_numpy()).sum())

    def group_weights(self, col: str, kinds: tuple[str, ...] = (CORP,)) -> pd.Series:
        f = self.risk.frame
        mask = f["kind"].isin(kinds).to_numpy()
        s = pd.Series(self.weights[mask], index=f.loc[mask, col].to_numpy())
        return s.groupby(level=0).sum().sort_values(ascending=False)

    @property
    def treasury_weight(self) -> float:
        return float(self.weights[(self.risk.frame["kind"] == TSY).to_numpy()].sum())

    def credit_exposure(self) -> pd.DataFrame:
        f = self.risk.frame
        mask = (f["kind"] == CORP).to_numpy()
        d = pd.DataFrame({"rating": f.loc[mask, "rating"].to_numpy(),
                          "market_value": self.market_values[mask],
                          "cs01": self.face[mask] / 100.0 * f.loc[mask, "cs01"].to_numpy()})
        g = d.groupby("rating").sum().reindex(["AA", "A", "BBB"]).fillna(0.0)
        g["weight"] = g["market_value"] / self.nav
        return g

    def ladder(self) -> pd.DataFrame:
        f = self.risk.frame
        bucket = pd.cut(f["years_to_maturity"], LADDER_EDGES, labels=LADDER_LABELS, right=False)
        d = pd.DataFrame({"bucket": bucket, "kind": f["kind"], "mv": self.market_values,
                          "principal": self.face, "dv01": self.face / 100.0 * f["dv01"].to_numpy()})
        out = d.pivot_table(index="bucket", columns="kind", values="mv", aggfunc="sum",
                            observed=False).reindex(columns=[TSY, CORP]).fillna(0.0)
        out.columns = ["treasury_mv", "corporate_mv"]
        out["principal"] = d.groupby("bucket", observed=False)["principal"].sum()
        out["dv01"] = d.groupby("bucket", observed=False)["dv01"].sum()
        out["weight"] = (out["treasury_mv"] + out["corporate_mv"]) / self.nav
        return out


def faces_from_weights(weights: np.ndarray, dirty: np.ndarray, nav: float,
                       rounding: float) -> np.ndarray:
    """Face amounts (rounded) that buy ``weights`` of ``nav`` at dirty prices per 100."""
    raw = weights * nav / (dirty / 100.0)
    return np.round(raw / rounding) * rounding if rounding > 0 else raw


def benchmark_weights(frame: pd.DataFrame, outstanding: np.ndarray, blocks: dict[str, float]) -> np.ndarray:
    mv = outstanding * frame["dirty"].to_numpy()
    w = np.zeros(len(frame))
    for kind, share in blocks.items():
        m = (frame["kind"] == kind).to_numpy()
        w[m] = share * mv[m] / mv[m].sum()
    return w
