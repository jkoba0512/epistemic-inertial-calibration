"""Planar 2-DoF (2R) rigid-body dynamics and inertial-parameter regressor (S0).

In line with the S0 goal of "fixing the coordinate system and the floor (base
parameters) of the identification problem". Physical-consistency constraints,
friction, Fourier excitation, etc. are intentionally NOT included (out of S0 scope).

Key design decision
-------------------
Rigid-body dynamics is linear in the standard inertial parameters, but the linearity
holds with respect to each link's *barycentric* parameters, not the raw (m, r, I)
(M11 contains m2*r2^2, which is nonlinear in (m2, m2*r2, I2)). We therefore adopt,
for a planar link whose CoM lies on the link axis (CoM distance r_i, inertia about
the CoM I_i), the standard linear parameters

    beta = [m1, h1, J1, m2, h2, J2]
    h_i = m_i * r_i             (first moment)
    J_i = I_i + m_i * r_i^2      (second moment about the joint origin)

Link length l1 and gravity g are treated as known constants.

The regressor Y(q, qd, qdd) is obtained by deriving the Lagrangian from the
kinematics with sympy, re-parameterizing the torque in the barycentric parameters,
and taking the Jacobian with respect to beta (so tau = Y @ beta holds exactly).
The raw m1 does not appear in the equations at all, so the m1 column of Y is
identically zero (the simplest demonstration of structural unidentifiability).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import sympy as sp

# Label order of beta (canonical ordering shared across all modules).
BETA_LABELS: tuple[str, ...] = ("m1", "h1", "J1", "m2", "h2", "J2")
N_BETA = len(BETA_LABELS)

G_DEFAULT = 9.81  # m/s^2, vertical-plane gravitational acceleration


@dataclass(frozen=True)
class Planar2RModel:
    """Known/unknown quantities of the 2R planar arm.

    l1: length of link 1 (known kinematic constant)
    g:  gravitational acceleration. 9.81 for the vertical plane, 0.0 for
        horizontal / gravity-free.
    True values (inertial parameters used for synthetic data / tests):
        m1, r1, I1, m2, r2, I2
    """

    l1: float = 1.0
    g: float = G_DEFAULT
    m1: float = 1.2
    r1: float = 0.45
    I1: float = 0.07
    m2: float = 0.9
    r2: float = 0.40
    I2: float = 0.05

    @property
    def gravity_in_plane(self) -> bool:
        return self.g != 0.0

    def beta_true(self) -> np.ndarray:
        """Build the barycentric beta from the true inertial parameters."""
        h1 = self.m1 * self.r1
        J1 = self.I1 + self.m1 * self.r1**2
        h2 = self.m2 * self.r2
        J2 = self.I2 + self.m2 * self.r2**2
        return np.array([self.m1, h1, J1, self.m2, h2, J2], dtype=float)


# ---------------------------------------------------------------------------
# Symbolic derivation (Lagrangian -> tau -> barycentric reparam -> regressor Y)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _symbolic() -> dict:
    """Derive the torque expression and regressor Y once from the Lagrangian; cached.

    Returns a dict with:
        beta_syms: sympy symbols (m1,h1,J1,m2,h2,J2)
        state_syms: (q1,q2,dq1,dq2,ddq1,ddq2)
        const_syms: (l1, g)
        tau: 2-vector torque expression (linear in beta)
        Y:   2x6 regressor (tau = Y @ beta)
    """
    t = sp.symbols("t", real=True)
    # Generalized coordinates as functions of time (so Euler-Lagrange can use d/dt).
    q1f = sp.Function("q1")(t)
    q2f = sp.Function("q2")(t)

    # Raw physical parameters.
    m1, r1, I1, m2, r2, I2 = sp.symbols("m1 r1 I1 m2 r2 I2", real=True, positive=True)
    l1, g = sp.symbols("l1 g", real=True)

    # Forward kinematics (CoM on the link axis). Base at origin, gravity along -y.
    xc1 = r1 * sp.cos(q1f)
    yc1 = r1 * sp.sin(q1f)
    xj2 = l1 * sp.cos(q1f)
    yj2 = l1 * sp.sin(q1f)
    xc2 = xj2 + r2 * sp.cos(q1f + q2f)
    yc2 = yj2 + r2 * sp.sin(q1f + q2f)

    vx1, vy1 = sp.diff(xc1, t), sp.diff(yc1, t)
    vx2, vy2 = sp.diff(xc2, t), sp.diff(yc2, t)
    w1 = sp.diff(q1f, t)
    w2 = sp.diff(q1f + q2f, t)

    # Kinetic and potential energy.
    T = (
        sp.Rational(1, 2) * m1 * (vx1**2 + vy1**2)
        + sp.Rational(1, 2) * I1 * w1**2
        + sp.Rational(1, 2) * m2 * (vx2**2 + vy2**2)
        + sp.Rational(1, 2) * I2 * w2**2
    )
    V = m1 * g * yc1 + m2 * g * yc2
    L = T - V

    # Euler-Lagrange: tau_i = d/dt(dL/dqdot_i) - dL/dq_i
    dq1 = sp.diff(q1f, t)
    dq2 = sp.diff(q2f, t)
    tau1 = sp.diff(sp.diff(L, dq1), t) - sp.diff(L, q1f)
    tau2 = sp.diff(sp.diff(L, dq2), t) - sp.diff(L, q2f)

    # Replace time functions with plain symbols.
    q1, q2, v1, v2, a1q, a2q = sp.symbols("q1 q2 dq1 dq2 ddq1 ddq2", real=True)
    subs_state = {
        sp.diff(q1f, t, t): a1q,
        sp.diff(q2f, t, t): a2q,
        sp.diff(q1f, t): v1,
        sp.diff(q2f, t): v2,
        q1f: q1,
        q2f: q2,
    }
    tau1 = sp.expand(tau1.subs(subs_state))
    tau2 = sp.expand(tau2.subs(subs_state))

    # Barycentric reparameterization:
    #   m1, h1=m1 r1, J1=I1+m1 r1^2, m2, h2=m2 r2, J2=I2+m2 r2^2
    # Substituting the inverse map r1=h1/m1, I1=J1-h1^2/m1, r2=h2/m2, I2=J2-h2^2/m2
    # cancels the nonlinear terms in m1, m2, leaving tau linear in beta.
    bm1, bh1, bJ1, bm2, bh2, bJ2 = sp.symbols("bm1 bh1 bJ1 bm2 bh2 bJ2", real=True)
    subs_bary = {
        r1: bh1 / bm1,
        I1: bJ1 - bh1**2 / bm1,
        r2: bh2 / bm2,
        I2: bJ2 - bh2**2 / bm2,
        m1: bm1,
        m2: bm2,
    }
    tau1 = sp.simplify(tau1.subs(subs_bary))
    tau2 = sp.simplify(tau2.subs(subs_bary))

    beta_syms = (bm1, bh1, bJ1, bm2, bh2, bJ2)
    tau_vec = sp.Matrix([tau1, tau2])
    Y = sp.simplify(tau_vec.jacobian(beta_syms))

    return {
        "beta_syms": beta_syms,
        "state_syms": (q1, q2, v1, v2, a1q, a2q),
        "const_syms": (l1, g),
        "tau": tau_vec,
        "Y": Y,
    }


@lru_cache(maxsize=8)
def _regressor_func(l1: float, g: float):
    """lambdify'd regressor Y(q1,q2,dq1,dq2,ddq1,ddq2) -> 2x6 for numeric eval."""
    sym = _symbolic()
    l1s, gs = sym["const_syms"]
    Y = sym["Y"].subs({l1s: sp.Float(l1), gs: sp.Float(g)})
    func = sp.lambdify(sym["state_syms"], Y, modules="numpy")
    return func


@lru_cache(maxsize=8)
def _tau_func(l1: float, g: float):
    """Independent numeric torque (Lagrangian tau lambdified with beta symbols)."""
    sym = _symbolic()
    l1s, gs = sym["const_syms"]
    tau = sym["tau"].subs({l1s: sp.Float(l1), gs: sp.Float(g)})
    args = list(sym["state_syms"]) + list(sym["beta_syms"])
    func = sp.lambdify(args, tau, modules="numpy")
    return func


def regressor(q: np.ndarray, qd: np.ndarray, qdd: np.ndarray, model: Planar2RModel) -> np.ndarray:
    """Return the regressor Y(q,qd,qdd) for a single sample (shape 2x6)."""
    f = _regressor_func(float(model.l1), float(model.g))
    Y = np.asarray(f(q[0], q[1], qd[0], qd[1], qdd[0], qdd[1]), dtype=float)
    return Y.reshape(2, N_BETA)


def inverse_dynamics(
    q: np.ndarray, qd: np.ndarray, qdd: np.ndarray, model: Planar2RModel
) -> np.ndarray:
    """Torque via the regressor: tau = Y(q,qd,qdd) @ beta_true."""
    return regressor(q, qd, qdd, model) @ model.beta_true()


def inverse_dynamics_closed_form(
    q: np.ndarray, qd: np.ndarray, qdd: np.ndarray, model: Planar2RModel
) -> np.ndarray:
    """Textbook closed-form 2R inverse dynamics (a path independent of the regressor).

    Spong & Vidyasagar form. Assembles M, C, g directly and returns the torque.
    Used to validate the regressor (consistency check).
    """
    m1, r1, I1 = model.m1, model.r1, model.I1
    m2, r2, I2 = model.m2, model.r2, model.I2
    l1, g = model.l1, model.g
    q1, q2 = q
    dq1, dq2 = qd
    c2, s2 = np.cos(q2), np.sin(q2)

    J1 = I1 + m1 * r1**2
    J2 = I2 + m2 * r2**2
    # Inertia matrix.
    M11 = J1 + J2 + m2 * l1**2 + 2.0 * m2 * l1 * r2 * c2
    M12 = J2 + m2 * l1 * r2 * c2
    M22 = J2
    M = np.array([[M11, M12], [M12, M22]])
    # Coriolis / centrifugal.
    h = m2 * l1 * r2 * s2
    c_vec = np.array([-h * (2.0 * dq1 * dq2 + dq2**2), h * dq1**2])
    # Gravity.
    g_vec = np.array(
        [
            (m1 * r1 + m2 * l1) * g * np.cos(q1) + m2 * r2 * g * np.cos(q1 + q2),
            m2 * r2 * g * np.cos(q1 + q2),
        ]
    )
    return M @ np.asarray(qdd, dtype=float) + c_vec + g_vec


def analytic_base_dim_symbolic(model: Planar2RModel) -> int:
    """Exact ground-truth base dimension via symbolic rank (no float threshold).

    The number of base parameters equals "the dimension of the function space
    spanned by the columns of the regressor Y as the state (q,qd,qdd) varies".
    To measure this exactly:
      1. expand_trig so cos(q1+q2) etc. expand into generators
         {cos q1, sin q1, cos q2, sin q2}.
      2. Convert each column into a coefficient vector over generators indexed by
         (row slot i, state monomial).
      3. Compute the rank of the span of the 6 coefficient vectors over the rationals.
    Functions like cos q1 and sin q1, and products like cos q1 * cos q2, are each
    treated as independent generators (functionally independent for generic angles).
    """
    sym = _symbolic()
    l1s, gs = sym["const_syms"]
    subs_c = {
        l1s: sp.nsimplify(model.l1, rational=True),
        gs: sp.nsimplify(model.g, rational=True),
    }
    Y = sym["Y"].subs(subs_c)
    state = sym["state_syms"]

    index: dict[tuple[int, sp.Expr], int] = {}
    cols: list[dict[int, sp.Expr]] = []
    for j in range(N_BETA):
        vec: dict[int, sp.Expr] = {}
        for i in range(2):
            expr = sp.expand(sp.expand_trig(Y[i, j]))
            if expr == 0:
                continue
            for term in expr.as_ordered_terms():
                coeff, monom = term.as_independent(*state, as_Add=False)
                key = (i, monom)
                idx = index.setdefault(key, len(index))
                vec[idx] = vec.get(idx, sp.Integer(0)) + coeff
        cols.append(vec)

    nrows = len(index)
    M = sp.zeros(nrows, N_BETA)
    for j, vec in enumerate(cols):
        for r, c in vec.items():
            M[r, j] = c
    return int(M.rank())


def stacked_regressor(samples: np.ndarray, model: Planar2RModel) -> np.ndarray:
    """Stack the regressor over a sample list (N x 6: q1,q2,dq1,dq2,ddq1,ddq2) -> W (2N x 6)."""
    rows = []
    for s in samples:
        q = s[0:2]
        qd = s[2:4]
        qdd = s[4:6]
        rows.append(regressor(q, qd, qdd, model))
    return np.vstack(rows)
