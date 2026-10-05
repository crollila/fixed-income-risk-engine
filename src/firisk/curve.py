"""Zero-coupon (spot) curve with log-linear discount-factor interpolation.

The curve stores continuously compounded zero rates at pillar times (ACT/365F
years). ``ln DF(t)`` is linear between pillars (and from the origin, where
DF = 1), so instantaneous forward rates are piecewise constant and discount
factors stay positive and, for positive forwards, decreasing. Beyond the last
pillar the last forward rate is extended (flat-forward extrapolation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ZeroCurve:
    times: np.ndarray  # pillar times, strictly increasing, > 0
    zeros: np.ndarray  # continuously compounded zero rates at pillars

    def __post_init__(self) -> None:
        t = np.asarray(self.times, dtype=float)
        z = np.asarray(self.zeros, dtype=float)
        if t.ndim != 1 or t.shape != z.shape or t.size == 0:
            raise ValueError("times and zeros must be equal-length 1-D arrays")
        if t[0] <= 0 or np.any(np.diff(t) <= 0):
            raise ValueError("pillar times must be positive and strictly increasing")
        object.__setattr__(self, "times", t)
        object.__setattr__(self, "zeros", z)

    @property
    def _nodes(self) -> tuple[np.ndarray, np.ndarray]:
        return np.concatenate([[0.0], self.times]), np.concatenate([[0.0], -self.zeros * self.times])

    def log_df(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        x, y = self._nodes
        out = np.interp(t, x, y)
        beyond = t > x[-1]
        if np.any(beyond):
            last_fwd = (y[-2] - y[-1]) / (x[-1] - x[-2])
            out = np.where(beyond, y[-1] - last_fwd * (t - x[-1]), out)
        return out

    def df(self, t) -> np.ndarray:
        return np.exp(self.log_df(t))

    def zero_rate(self, t, comp: str = "cont") -> np.ndarray:
        t = np.asarray(t, dtype=float)
        r = -self.log_df(t) / t
        return convert_from_cont(r, comp)

    def forward_rate(self, t1, t2, comp: str = "cont") -> np.ndarray:
        """Forward rate between t1 and t2 (continuous, or ``simple`` / ``semi``)."""
        t1 = np.asarray(t1, dtype=float)
        t2 = np.asarray(t2, dtype=float)
        r = (self.log_df(t1) - self.log_df(t2)) / (t2 - t1)
        if comp == "simple":
            return (np.exp(r * (t2 - t1)) - 1.0) / (t2 - t1)
        return convert_from_cont(r, comp)

    def instantaneous_forward(self, t) -> np.ndarray:
        return self.forward_rate(np.asarray(t) - 1e-6, np.asarray(t) + 1e-6)


def convert_from_cont(r, comp: str):
    if comp == "cont":
        return r
    if comp == "semi":
        return 2.0 * (np.exp(r / 2.0) - 1.0)
    if comp == "annual":
        return np.exp(r) - 1.0
    raise ValueError(f"unknown compounding {comp!r}")


def flat_curve(rate_cont: float, max_t: float = 50.0) -> ZeroCurve:
    return ZeroCurve(np.array([max_t]), np.array([rate_cont]))
