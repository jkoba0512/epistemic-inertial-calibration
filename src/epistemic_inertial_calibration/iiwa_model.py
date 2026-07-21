"""S4-0: KUKA LBR iiwa14 (7-DoF) model wrapper over Pinocchio.

Generalizes the planar dynamics/regressor/control pipeline to a 7-DoF manipulator
for the S4 case study. Requires the optional `iiwa` extra (`uv sync --extra iiwa`);
importing this module without Pinocchio raises ImportError, and the S4 tests guard
with pytest.importorskip so the S0-S3 mainline is unaffected.

The vendored URDF (assets/iiwa/, BSD-3-Clause) is loaded model-only (no meshes).
Pinocchio provides everything we need:
  - rnea  : inverse dynamics (torque)
  - crba  : joint-space mass matrix M(q)
  - aba   : forward dynamics (qddot)  -> closed-loop rollout plant
  - computeJointTorqueRegressor : the standard inertial-parameter regressor
    Y(q,qd,qdd) with tau = Y @ phi, phi = 10 params per link (70 total)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import pinocchio as pin
except ImportError as e:  # pragma: no cover - exercised only without the iiwa extra
    raise ImportError(
        "iiwa_model requires Pinocchio. Install the optional extra: `uv sync --extra iiwa`."
    ) from e

# Vendored URDF (repo self-contained; see assets/iiwa/README.md).
DEFAULT_URDF = Path(__file__).resolve().parents[2] / "assets" / "iiwa" / "iiwa14_no_collision.urdf"


@dataclass
class IiwaModel:
    """Thin wrapper bundling a Pinocchio model + data and convenience methods."""

    model: "pin.Model"
    data: "pin.Data"
    ee_frame_id: int

    @property
    def nq(self) -> int:
        return self.model.nq

    @property
    def nv(self) -> int:
        return self.model.nv

    @property
    def n_params(self) -> int:
        return 10 * (self.model.njoints - 1)  # 10 inertial params per moving link

    def effort_limit(self) -> np.ndarray:
        return np.array(self.model.effortLimit, dtype=float)

    def beta_true(self) -> np.ndarray:
        """Standard inertial parameter vector phi (10 per link) from the URDF."""
        return np.concatenate(
            [self.model.inertias[i].toDynamicParameters() for i in range(1, self.model.njoints)]
        )

    # --- kinematics --------------------------------------------------------

    def forward_kinematics(self, q) -> np.ndarray:
        q = np.asarray(q, float)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        return np.array(self.data.oMf[self.ee_frame_id].translation, dtype=float)

    def ee_pose(self, q):
        q = np.asarray(q, float)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        return self.data.oMf[self.ee_frame_id]

    def jacobian(self, q) -> np.ndarray:
        """Spatial (6 x nv) end-effector Jacobian in the local-world-aligned frame."""
        q = np.asarray(q, float)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return np.array(
            pin.getFrameJacobian(self.model, self.data, self.ee_frame_id,
                                 pin.ReferenceFrame.LOCAL_WORLD_ALIGNED),
            dtype=float,
        )

    def position_jacobian(self, q) -> np.ndarray:
        """3 x nv translational EE Jacobian (top three rows)."""
        return self.jacobian(q)[:3, :]

    # --- dynamics ----------------------------------------------------------

    def inverse_dynamics(self, q, qd, qdd) -> np.ndarray:
        return np.array(pin.rnea(self.model, self.data,
                                 np.asarray(q, float), np.asarray(qd, float),
                                 np.asarray(qdd, float)), dtype=float)

    def mass_matrix(self, q) -> np.ndarray:
        M = np.array(pin.crba(self.model, self.data, np.asarray(q, float)), dtype=float)
        return np.triu(M) + np.triu(M, 1).T  # crba fills upper triangle only

    def forward_dynamics(self, q, qd, tau) -> np.ndarray:
        return np.array(pin.aba(self.model, self.data,
                                np.asarray(q, float), np.asarray(qd, float),
                                np.asarray(tau, float)), dtype=float)

    def regressor(self, q, qd, qdd) -> np.ndarray:
        """Standard inertial-parameter regressor Y(q,qd,qdd), shape (nv, 10*nlinks)."""
        return np.array(
            pin.computeJointTorqueRegressor(self.model, self.data,
                                            np.asarray(q, float), np.asarray(qd, float),
                                            np.asarray(qdd, float)),
            dtype=float,
        )

    def random_configuration(self, rng: np.random.Generator | None = None) -> np.ndarray:
        if rng is None:
            return np.array(pin.randomConfiguration(self.model), dtype=float)
        lo = np.array(self.model.lowerPositionLimit, dtype=float)
        hi = np.array(self.model.upperPositionLimit, dtype=float)
        lo = np.where(np.isfinite(lo), lo, -np.pi)
        hi = np.where(np.isfinite(hi), hi, np.pi)
        return rng.uniform(lo, hi)


def load_iiwa(urdf_path: str | Path = DEFAULT_URDF, ee_frame: str = "iiwa_link_ee") -> IiwaModel:
    """Load the vendored iiwa14 model (model only, no geometry)."""
    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    # Resolve an end-effector frame; fall back to the last frame if the name is absent.
    if model.existFrame(ee_frame):
        fid = model.getFrameId(ee_frame)
    else:
        fid = model.nframes - 1
    return IiwaModel(model=model, data=data, ee_frame_id=fid)
