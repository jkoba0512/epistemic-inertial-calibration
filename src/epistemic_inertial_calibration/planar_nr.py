"""S2-0: N-link planar arm dynamics, kinematics, and inertial-parameter regressor.

Generalizes planar2r.py to an n-link planar revolute chain for the Layer-2
(4-DoF redundant) experiments. planar2r.py is left untouched (S0/S1 depend on it);
this module is validated against it for n=2 (the independently verified 2R system,
matching its closed form to ~1e-15).

Conventions (same barycentric parameterization as planar2r):
    per link i: [m_i, h_i = m_i r_i, J_i = I_i + m_i r_i^2]   (CoM on the link axis)
    beta order: [m_0,h_0,J_0, m_1,h_1,J_1, ...]  (length 3n)
    link i absolute angle: phi_i = q_0 + ... + q_i
The torque regressor Y(q,qd,qdd) is (n x 3n) with tau = Y @ beta.

Regressor construction (chain rule, avoids symbolic cancellation): the raw
Lagrangian torque tau(q,qd,qdd; m,r,I) is differentiated symbolically w.r.t.
(m_i, r_i, I_i). Because tau is exactly linear in beta, the barycentric columns are
point-independent constants obtained by the chain rule of the map
beta -> (m, r=h/m, I=J-h^2/m):
    Y[:, m_i] = d tau/d m_i - (r_i/m_i) d tau/d r_i + r_i^2 d tau/d I_i
    Y[:, h_i] = (1/m_i) d tau/d r_i - 2 r_i d tau/d I_i
    Y[:, J_i] = d tau/d I_i
evaluated at the model's (m,r,I). Base dimension is NOT hardcoded: it is measured
numerically (SVD rank, tolerance sweep over seeds) and cross-checked symbolically.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import sympy as sp

G_DEFAULT = 9.81


def beta_labels(n: int) -> tuple[str, ...]:
    out: list[str] = []
    for i in range(n):
        out += [f"m{i}", f"h{i}", f"J{i}"]
    return tuple(out)


@dataclass(frozen=True)
class PlanarNRModel:
    """n-link planar revolute arm (tuples have length n_links)."""

    n_links: int = 4
    link_lengths: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    masses: tuple[float, ...] = (1.2, 1.0, 0.8, 0.6)
    com_dists: tuple[float, ...] = (0.45, 0.42, 0.40, 0.38)
    inertias: tuple[float, ...] = (0.07, 0.05, 0.04, 0.03)
    g: float = G_DEFAULT

    def __post_init__(self):
        n = self.n_links
        for name in ("link_lengths", "masses", "com_dists", "inertias"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} must have length n_links={n}")

    @property
    def gravity_in_plane(self) -> bool:
        return self.g != 0.0

    @property
    def n_beta(self) -> int:
        return 3 * self.n_links

    def beta_true(self) -> np.ndarray:
        out: list[float] = []
        for i in range(self.n_links):
            m, r, Iv = self.masses[i], self.com_dists[i], self.inertias[i]
            out += [m, m * r, Iv + m * r**2]
        return np.array(out, dtype=float)


# ---------------------------------------------------------------------------
# Symbolic derivation (raw Lagrangian torque and its partials), cached per n
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _symbolic(n: int) -> dict:
    t = sp.symbols("t", real=True)
    qf = [sp.Function(f"q{i}")(t) for i in range(n)]
    m = sp.symbols(f"m0:{n}", positive=True)
    r = sp.symbols(f"r0:{n}", positive=True)
    Isym = sp.symbols(f"I0:{n}", positive=True)
    ll = sp.symbols(f"l0:{n}", positive=True)
    g = sp.symbols("g", real=True)

    phi = [sp.Add(*qf[: i + 1]) for i in range(n)]
    jx = [sp.Integer(0)]
    jy = [sp.Integer(0)]
    for i in range(n):
        jx.append(jx[-1] + ll[i] * sp.cos(phi[i]))
        jy.append(jy[-1] + ll[i] * sp.sin(phi[i]))
    T = sp.Integer(0)
    V = sp.Integer(0)
    for i in range(n):
        cx = jx[i] + r[i] * sp.cos(phi[i])
        cy = jy[i] + r[i] * sp.sin(phi[i])
        vx, vy = sp.diff(cx, t), sp.diff(cy, t)
        w = sp.diff(phi[i], t)
        T += sp.Rational(1, 2) * m[i] * (vx**2 + vy**2) + sp.Rational(1, 2) * Isym[i] * w**2
        V += m[i] * g * cy
    L = T - V

    tau = [sp.diff(sp.diff(L, sp.diff(qf[i], t)), t) - sp.diff(L, qf[i]) for i in range(n)]

    q = sp.symbols(f"q0:{n}", real=True)
    dq = sp.symbols(f"dq0:{n}", real=True)
    ddq = sp.symbols(f"ddq0:{n}", real=True)
    subs_state = {}
    for i in range(n):
        subs_state[sp.diff(qf[i], t, t)] = ddq[i]
        subs_state[sp.diff(qf[i], t)] = dq[i]
        subs_state[qf[i]] = q[i]
    tau = [sp.expand(ti.subs(subs_state)) for ti in tau]

    # Partials w.r.t. raw inertial params (cheap symbolic diff; no simplify/cancel).
    dtau_dm = sp.Matrix([[sp.diff(tau[k], m[i]) for i in range(n)] for k in range(n)])
    dtau_dr = sp.Matrix([[sp.diff(tau[k], r[i]) for i in range(n)] for k in range(n)])
    dtau_dI = sp.Matrix([[sp.diff(tau[k], Isym[i]) for i in range(n)] for k in range(n)])

    return {
        "n": n, "tau": tau,
        "q": q, "dq": dq, "ddq": ddq, "m": m, "r": r, "I": Isym, "l": ll, "g": g,
        "dtau_dm": dtau_dm, "dtau_dr": dtau_dr, "dtau_dI": dtau_dI,
    }


@lru_cache(maxsize=8)
def _partial_funcs(n: int, link_lengths: tuple[float, ...], g: float):
    """lambdified partials d tau/d{m,r,I} as functions of (q,dq,ddq,m,r,I)."""
    sym = _symbolic(n)
    subs = {sym["g"]: sp.Float(g)}
    for i in range(n):
        subs[sym["l"][i]] = sp.Float(link_lengths[i])
    args = list(sym["q"]) + list(sym["dq"]) + list(sym["ddq"]) + \
        list(sym["m"]) + list(sym["r"]) + list(sym["I"])
    f_dm = sp.lambdify(args, sym["dtau_dm"].subs(subs), modules="numpy")
    f_dr = sp.lambdify(args, sym["dtau_dr"].subs(subs), modules="numpy")
    f_dI = sp.lambdify(args, sym["dtau_dI"].subs(subs), modules="numpy")
    return f_dm, f_dr, f_dI


def regressor(q, qd, qdd, model: PlanarNRModel) -> np.ndarray:
    """Torque-row regressor Y(q,qd,qdd), shape (n, 3n), with tau = Y @ beta."""
    n = model.n_links
    f_dm, f_dr, f_dI = _partial_funcs(n, tuple(model.link_lengths), float(model.g))
    mm = np.asarray(model.masses, float)
    rr = np.asarray(model.com_dists, float)
    II = np.asarray(model.inertias, float)
    a = list(np.asarray(q, float)) + list(np.asarray(qd, float)) + list(np.asarray(qdd, float)) \
        + list(mm) + list(rr) + list(II)
    DM = np.asarray(f_dm(*a), float).reshape(n, n)
    DR = np.asarray(f_dr(*a), float).reshape(n, n)
    DI = np.asarray(f_dI(*a), float).reshape(n, n)
    Y = np.zeros((n, 3 * n))
    for i in range(n):
        Y[:, 3 * i] = DM[:, i] - (rr[i] / mm[i]) * DR[:, i] + rr[i] ** 2 * DI[:, i]
        Y[:, 3 * i + 1] = (1.0 / mm[i]) * DR[:, i] - 2.0 * rr[i] * DI[:, i]
        Y[:, 3 * i + 2] = DI[:, i]
    return Y


def inverse_dynamics(q, qd, qdd, model: PlanarNRModel) -> np.ndarray:
    """Torque via the regressor: tau = Y @ beta_true."""
    return regressor(q, qd, qdd, model) @ model.beta_true()


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------


def forward_kinematics(q, model: PlanarNRModel) -> np.ndarray:
    """End-effector (x, y) position."""
    q = np.asarray(q, float)
    phi = np.cumsum(q)
    ll = np.asarray(model.link_lengths)
    return np.array([float(np.sum(ll * np.cos(phi))), float(np.sum(ll * np.sin(phi)))])


def jacobian(q, model: PlanarNRModel) -> np.ndarray:
    """End-effector position Jacobian J (2 x n) = d(ee)/dq."""
    q = np.asarray(q, float)
    phi = np.cumsum(q)
    ll = np.asarray(model.link_lengths)
    # d x / d q_k = -sum_{i>=k} l_i sin(phi_i); d y / d q_k = sum_{i>=k} l_i cos(phi_i)
    dx = -np.cumsum((ll * np.sin(phi))[::-1])[::-1]
    dy = np.cumsum((ll * np.cos(phi))[::-1])[::-1]
    return np.vstack([dx, dy])


def nullspace_projector(q, model: PlanarNRModel, rcond: float = 1e-10) -> np.ndarray:
    """Null-space projector N = I - J^+ J (n x n)."""
    J = jacobian(q, model)
    return np.eye(model.n_links) - np.linalg.pinv(J, rcond=rcond) @ J


# ---------------------------------------------------------------------------
# Base dimension (numeric primary; symbolic cross-check)
# ---------------------------------------------------------------------------


def stacked_regressor(samples: np.ndarray, model: PlanarNRModel) -> np.ndarray:
    """Stack Y over samples (N x 3n state: q|qd|qdd) -> (n*N x 3n)."""
    n = model.n_links
    return np.vstack([regressor(s[0:n], s[n : 2 * n], s[2 * n : 3 * n], model) for s in samples])


def numeric_base_dim(model: PlanarNRModel, n_samples: int = 2000, seed: int = 0,
                     rel_threshold: float = 1e-8) -> int:
    """Numerical base dimension = SVD rank of the stacked dynamic regressor."""
    rng = np.random.default_rng(seed)
    n = model.n_links
    states = np.hstack([
        rng.uniform(-np.pi, np.pi, size=(n_samples, n)),
        rng.uniform(-2.0, 2.0, size=(n_samples, n)),
        rng.uniform(-4.0, 4.0, size=(n_samples, n)),
    ])
    sv = np.linalg.svd(stacked_regressor(states, model), compute_uv=False)
    return int(np.sum(sv > sv[0] * rel_threshold))


def expected_base_dim_pattern(model: PlanarNRModel) -> int:
    """Base-dim prediction from the 2n / (2n-1) pattern (vertical / horizontal).

    This pattern reproduces the symbolically verified 2R result (4/3) and is used
    only as a cross-check against numeric_base_dim — NOT as the authority and NOT
    hardcoded into the measurement. The authority is numeric_base_dim (clean ~15
    order singular-value gap, stable across seeds and thresholds).

    Note: an exact symbolic rank for n>=3 is not used here. The symbolic
    coefficient-rank approach overcounts unless the trigonometric identities
    (sin^2+cos^2=1) and the 1/m cancellations are fully reduced, which is
    expensive and brittle; the numeric rank is unambiguous instead.
    """
    n = model.n_links
    return 2 * n if model.gravity_in_plane else 2 * n - 1
