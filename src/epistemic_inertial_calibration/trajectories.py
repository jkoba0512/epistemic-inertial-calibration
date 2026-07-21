"""S1B: state-window generators for the excitation policies.

A "window" is an array of states (n x 6: q1,q2,dq1,dq2,ddq1,ddq2). Policies build a
dataset by selecting one window per round. All randomness flows through an explicit
numpy Generator so candidate sets can be made reproducible and shared across
policies (e.g. active_ig vs ig_fixed_prior).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindowConfig:
    """Scales/shape for window sampling."""

    q_range: float = np.pi
    vel_scale: float = 2.0
    acc_scale: float = 4.0
    quasi_scale: float = 1e-3
    fourier_modes: int = 5
    fourier_base_freq: float = 1.0
    fourier_amp: float = 1.0
    fourier_horizon: float = 2.0


def sample_window(rng: np.random.Generator, regime: str, n: int, cfg: WindowConfig) -> np.ndarray:
    """Draw one window of n states for a regime, using the given rng."""
    q = rng.uniform(-cfg.q_range, cfg.q_range, size=(n, 2))
    if regime == "static":
        qd = np.zeros((n, 2))
        qdd = np.zeros((n, 2))
    elif regime == "quasi_static":
        qd = rng.normal(0.0, cfg.quasi_scale, size=(n, 2))
        qdd = rng.normal(0.0, cfg.quasi_scale, size=(n, 2))
    elif regime == "dynamic":
        qd = rng.uniform(-cfg.vel_scale, cfg.vel_scale, size=(n, 2))
        qdd = rng.uniform(-cfg.acc_scale, cfg.acc_scale, size=(n, 2))
    else:
        raise ValueError(f"unknown regime: {regime!r}")
    return np.hstack([q, qd, qdd])


def fourier_window(rng: np.random.Generator, n: int, cfg: WindowConfig) -> np.ndarray:
    """Generate a window from a scripted finite-Fourier-series trajectory.

    q_j(t) = sum_{k=1}^{K} [ a_jk sin(k w t) + b_jk cos(k w t) ] / k
    with qd, qdd obtained by exact analytic differentiation. This is the simple
    "broad excitation" classical baseline; D-optimal coefficient optimization is
    deferred (S1+ per the plan).
    """
    K = cfg.fourier_modes
    w = 2.0 * np.pi * cfg.fourier_base_freq
    t = np.linspace(0.0, cfg.fourier_horizon, n)
    a = rng.normal(0.0, cfg.fourier_amp, size=(2, K))
    b = rng.normal(0.0, cfg.fourier_amp, size=(2, K))

    q = np.zeros((n, 2))
    qd = np.zeros((n, 2))
    qdd = np.zeros((n, 2))
    for k in range(1, K + 1):
        wk = k * w
        sk = np.sin(wk * t)  # (n,)
        ck = np.cos(wk * t)
        for j in range(2):
            ajk = a[j, k - 1] / k
            bjk = b[j, k - 1] / k
            q[:, j] += ajk * sk + bjk * ck
            qd[:, j] += ajk * wk * ck - bjk * wk * sk
            qdd[:, j] += -ajk * wk**2 * sk - bjk * wk**2 * ck
    return np.hstack([q, qd, qdd])
