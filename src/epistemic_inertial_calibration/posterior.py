"""S1A: Gaussian linear-Gaussian posterior over base-coordinate parameters.

Model (per the S1 plan, 2026-06-18):
    prior:       alpha ~ N(mu0, P0)
    observation: y = W_base @ alpha + eps,  eps ~ N(0, sigma_tau^2 I)

Maintained in information (natural-parameter) form so that batches of data can be
added incrementally:
    Lambda = P0^{-1} + (1/sigma_tau^2) sum_k W_base_k^T W_base_k
    nu     = P0^{-1} mu0 + (1/sigma_tau^2) sum_k W_base_k^T y_k
    mu_N   = Lambda^{-1} nu
    P_N    = Lambda^{-1}

Adding data only adds PSD terms to Lambda, so the covariance is monotonically
non-increasing in the Loewner order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianPosterior:
    """Information-form posterior over alpha (base coordinates)."""

    Lambda: np.ndarray  # precision (dim x dim)
    nu: np.ndarray  # information vector = Lambda @ mu
    dim: int

    @classmethod
    def from_prior(cls, mu0: np.ndarray, P0: np.ndarray) -> "GaussianPosterior":
        mu0 = np.asarray(mu0, dtype=float)
        P0 = np.asarray(P0, dtype=float)
        Lambda0 = np.linalg.inv(P0)
        return cls(Lambda=Lambda0.copy(), nu=Lambda0 @ mu0, dim=mu0.size)

    @classmethod
    def isotropic_prior(cls, mu0: np.ndarray, prior_std: float) -> "GaussianPosterior":
        mu0 = np.asarray(mu0, dtype=float)
        P0 = (prior_std**2) * np.eye(mu0.size)
        return cls.from_prior(mu0, P0)

    def update(self, W_base: np.ndarray, y: np.ndarray, sigma_tau: float) -> None:
        """Incorporate a batch of observations (in place)."""
        inv_var = 1.0 / (sigma_tau**2)
        self.Lambda = self.Lambda + inv_var * (W_base.T @ W_base)
        self.nu = self.nu + inv_var * (W_base.T @ y)

    # --- queries ---------------------------------------------------------

    def covariance(self) -> np.ndarray:
        return np.linalg.inv(self.Lambda)

    def mean(self) -> np.ndarray:
        return np.linalg.solve(self.Lambda, self.nu)

    def logdet_cov(self) -> float:
        """log det of the covariance = -log det of the precision."""
        sign, logabsdet = np.linalg.slogdet(self.Lambda)
        if sign <= 0:
            return float("inf")
        return float(-logabsdet)

    def trace_cov(self) -> float:
        return float(np.trace(self.covariance()))

    def copy(self) -> "GaussianPosterior":
        return GaussianPosterior(self.Lambda.copy(), self.nu.copy(), self.dim)


def information_gain(Lambda: np.ndarray, F: np.ndarray) -> float:
    """Expected information gain of adding Fisher information F to precision Lambda.

    IG = 0.5 [ log det(Lambda + F) - log det(Lambda) ].
    For the linear-Gaussian model, F = (1/sigma_tau^2) W_base(u)^T W_base(u).
    """
    s0, ld0 = np.linalg.slogdet(Lambda)
    s1, ld1 = np.linalg.slogdet(Lambda + F)
    if s0 <= 0 or s1 <= 0:
        return float("nan")
    return 0.5 * float(ld1 - ld0)


def fisher_information(W_base: np.ndarray, sigma_tau: float) -> np.ndarray:
    """Fisher information of a base-coordinate observation batch: (1/sigma^2) Wb^T Wb."""
    return (1.0 / (sigma_tau**2)) * (W_base.T @ W_base)
