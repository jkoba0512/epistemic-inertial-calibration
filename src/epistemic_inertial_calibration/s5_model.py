"""S5-0: friction-extended 2R model for the hardware-executable pipeline.

S5 rebuilds the identification pipeline so that every step is structurally
identical to a hardware experiment: excitation is a continuous executable
trajectory, observations come from a closed-loop rollout on the simulated
plant, and the observation model includes joint friction -- the dominant
non-rigid-body effect in real joint-torque data.

Friction model (per joint i):

    tau_fric_i = Fv_i * qd_i + Fc_i * tanh(qd_i / eps)

Viscous plus smoothed Coulomb friction. The tanh smoothing (width eps) keeps
the plant integrable and the regressor consistent with the plant; static
friction (stiction) below the smoothing width is intentionally not modeled.
The same smoothed model is used by the plant and by the regressor, so torque
remains exactly linear in the extended parameter vector

    beta_ext = [m1, h1, J1, m2, h2, J2, Fv1, Fc1, Fv2, Fc2].

The rigid-body block reuses the verified planar2r regressor; the legacy
6-parameter API is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .controllers import mcg_2r
from .planar2r import BETA_LABELS, Planar2RModel, regressor

FRICTION_LABELS: tuple[str, ...] = ("Fv1", "Fc1", "Fv2", "Fc2")
BETA_EXT_LABELS: tuple[str, ...] = BETA_LABELS + FRICTION_LABELS
N_BETA_EXT = len(BETA_EXT_LABELS)  # 10


@dataclass(frozen=True)
class FrictionModel2R:
    """2R model with viscous + smoothed-Coulomb joint friction.

    rigid: the rigid-body model (kinematics, gravity, true inertial params).
    fv, fc: true per-joint viscous / Coulomb coefficients (N m s/rad, N m).
    eps: Coulomb smoothing width (rad/s); shared by plant and regressor.
    """

    rigid: Planar2RModel = Planar2RModel()
    fv: tuple[float, float] = (0.40, 0.25)
    fc: tuple[float, float] = (0.80, 0.50)
    eps: float = 0.05

    @property
    def l1(self) -> float:
        return self.rigid.l1

    @property
    def g(self) -> float:
        return self.rigid.g

    def beta_ext_true(self) -> np.ndarray:
        """Extended true parameter vector [beta_rigid | Fv1, Fc1, Fv2, Fc2]."""
        fr = np.array([self.fv[0], self.fc[0], self.fv[1], self.fc[1]], dtype=float)
        return np.concatenate([self.rigid.beta_true(), fr])

    def friction_torque(self, qd: np.ndarray) -> np.ndarray:
        """tau_fric(qd) with the true coefficients (used by the plant)."""
        qd = np.asarray(qd, dtype=float)
        fv = np.array(self.fv, dtype=float)
        fc = np.array(self.fc, dtype=float)
        return fv * qd + fc * np.tanh(qd / self.eps)


def friction_regressor(qd: np.ndarray, eps: float) -> np.ndarray:
    """Friction block of the extended regressor (2 x 4).

    Row i has [qd_i, tanh(qd_i/eps)] in its own column pair and zeros in the
    other joint's pair, matching the FRICTION_LABELS ordering.
    """
    qd = np.asarray(qd, dtype=float)
    F = np.zeros((2, 4))
    F[0, 0] = qd[0]
    F[0, 1] = np.tanh(qd[0] / eps)
    F[1, 2] = qd[1]
    F[1, 3] = np.tanh(qd[1] / eps)
    return F


def regressor_ext(
    q: np.ndarray, qd: np.ndarray, qdd: np.ndarray, fmodel: FrictionModel2R
) -> np.ndarray:
    """Extended regressor Y_ext(q,qd,qdd) with tau = Y_ext @ beta_ext (2 x 10)."""
    Y = regressor(q, qd, qdd, fmodel.rigid)
    return np.hstack([Y, friction_regressor(qd, fmodel.eps)])


def inverse_dynamics_ext(
    q: np.ndarray, qd: np.ndarray, qdd: np.ndarray, fmodel: FrictionModel2R
) -> np.ndarray:
    """Torque via the extended regressor: tau = Y_ext @ beta_ext_true."""
    return regressor_ext(q, qd, qdd, fmodel) @ fmodel.beta_ext_true()


def forward_dynamics_ext(
    q: np.ndarray, qd: np.ndarray, tau: np.ndarray, fmodel: FrictionModel2R
) -> np.ndarray:
    """qdd = M^{-1} (tau - C qd - g - tau_fric), plant side of the friction model."""
    beta = fmodel.rigid.beta_true()
    M, c_vec, g_vec = mcg_2r(q, qd, beta, fmodel.l1, fmodel.g)
    rhs = np.asarray(tau, dtype=float) - c_vec - g_vec - fmodel.friction_torque(qd)
    return np.linalg.solve(M, rhs)


def stacked_regressor_ext(samples: np.ndarray, fmodel: FrictionModel2R) -> np.ndarray:
    """Stack Y_ext over samples (N x 6 states) -> (2N x 10)."""
    rows = [regressor_ext(s[0:2], s[2:4], s[4:6], fmodel) for s in samples]
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# Extended base projection artifact (analogous to the S0 artifact, 10 params)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class S5ProjectionArtifact:
    """Loaded S5 base-projection artifact (frozen input for the S5 pipeline)."""

    V_base: np.ndarray  # (rank x 10)
    rank: int
    beta_labels: tuple[str, ...]
    gravity_in_plane: bool
    l1: float
    g: float
    fv: tuple[float, float]
    fc: tuple[float, float]
    eps: float
    rel_threshold: float
    source_path: str

    def model(self) -> FrictionModel2R:
        """Reconstruct the FrictionModel2R matching this projection."""
        return FrictionModel2R(
            rigid=Planar2RModel(g=self.g, l1=self.l1), fv=self.fv, fc=self.fc, eps=self.eps
        )


def save_s5_projection(
    npz_path: str | Path,
    V_base: np.ndarray,
    rank: int,
    fmodel: FrictionModel2R,
    rel_threshold: float,
) -> None:
    np.savez(
        Path(npz_path),
        V_base=np.asarray(V_base, dtype=float),
        rank=int(rank),
        beta_labels=np.array(BETA_EXT_LABELS),
        gravity_in_plane=bool(fmodel.rigid.gravity_in_plane),
        l1=float(fmodel.l1),
        g=float(fmodel.g),
        fv=np.array(fmodel.fv, dtype=float),
        fc=np.array(fmodel.fc, dtype=float),
        eps=float(fmodel.eps),
        rel_threshold=float(rel_threshold),
    )


def load_s5_projection(npz_path: str | Path) -> S5ProjectionArtifact:
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as d:
        return S5ProjectionArtifact(
            V_base=np.asarray(d["V_base"], dtype=float),
            rank=int(d["rank"]),
            beta_labels=tuple(str(x) for x in d["beta_labels"]),
            gravity_in_plane=bool(d["gravity_in_plane"]),
            l1=float(d["l1"]),
            g=float(d["g"]),
            fv=(float(d["fv"][0]), float(d["fv"][1])),
            fc=(float(d["fc"][0]), float(d["fc"][1])),
            eps=float(d["eps"]),
            rel_threshold=float(d["rel_threshold"]),
            source_path=str(npz_path),
        )


def alpha_true_ext(proj: S5ProjectionArtifact, fmodel: FrictionModel2R) -> np.ndarray:
    """Identifiable target in extended base coordinates."""
    return proj.V_base @ fmodel.beta_ext_true()
