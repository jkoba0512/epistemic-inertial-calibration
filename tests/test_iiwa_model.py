"""S4-0: smoke tests for the iiwa14 Pinocchio wrapper.

Guarded by importorskip so the S0-S3 mainline stays green without the iiwa extra
(`uv sync --extra iiwa`).
"""

import numpy as np
import pytest

pytest.importorskip("pinocchio", reason="iiwa extra not installed (uv sync --extra iiwa)")

from epistemic_inertial_calibration.iiwa_model import DEFAULT_URDF, load_iiwa  # noqa: E402

pytestmark = pytest.mark.skipif(not DEFAULT_URDF.exists(), reason="vendored iiwa URDF missing")


def _model():
    return load_iiwa()


def test_load_and_dof():
    r = _model()
    assert r.nq == 7 and r.nv == 7
    assert r.n_params == 70  # 10 inertial params x 7 links


def test_forward_kinematics_and_jacobian():
    r = _model()
    rng = np.random.default_rng(0)
    q = r.random_configuration(rng)
    p = r.forward_kinematics(q)
    assert p.shape == (3,) and np.all(np.isfinite(p))
    assert r.jacobian(q).shape == (6, 7)
    assert r.position_jacobian(q).shape == (3, 7)


def test_jacobian_matches_finite_difference():
    r = _model()
    rng = np.random.default_rng(1)
    q = r.random_configuration(rng)
    Jp = r.position_jacobian(q)
    eps = 1e-6
    Jfd = np.zeros((3, 7))
    for k in range(7):
        d = np.zeros(7)
        d[k] = eps
        Jfd[:, k] = (r.forward_kinematics(q + d) - r.forward_kinematics(q - d)) / (2 * eps)
    assert np.allclose(Jp, Jfd, atol=1e-5)


def test_mass_matrix_spd():
    r = _model()
    rng = np.random.default_rng(2)
    for _ in range(20):
        M = r.mass_matrix(r.random_configuration(rng))
        assert np.allclose(M, M.T)
        assert np.all(np.linalg.eigvalsh(M) > 0)


def test_forward_inverse_roundtrip():
    r = _model()
    rng = np.random.default_rng(3)
    max_err = 0.0
    for _ in range(50):
        q = r.random_configuration(rng)
        qd = rng.normal(size=7)
        qdd = rng.normal(size=7)
        tau = r.inverse_dynamics(q, qd, qdd)
        qdd_back = r.forward_dynamics(q, qd, tau)
        max_err = max(max_err, np.max(np.abs(qdd - qdd_back)))
    assert max_err < 1e-7


def test_regressor_matches_inverse_dynamics():
    """Y(q,qd,qdd) @ beta_true == rnea torque (standard inertial regressor identity)."""
    r = _model()
    beta = r.beta_true()
    rng = np.random.default_rng(4)
    max_rel = 0.0
    for _ in range(50):
        q = r.random_configuration(rng)
        qd = rng.normal(size=7)
        qdd = rng.normal(size=7)
        Y = r.regressor(q, qd, qdd)
        assert Y.shape == (7, 70)
        tau = r.inverse_dynamics(q, qd, qdd)
        max_rel = max(max_rel, np.max(np.abs(Y @ beta - tau)) / (np.max(np.abs(tau)) + 1e-9))
    assert max_rel < 1e-7


def test_effort_limits_defined():
    r = _model()
    el = r.effort_limit()
    assert el.shape == (7,)
    assert np.all(el > 0) and np.all(np.isfinite(el))
