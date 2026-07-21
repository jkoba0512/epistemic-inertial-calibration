"""S2-0: tests for the N-link planar model, kinematics, and base dimension.

The general n-link regressor is validated against the independently verified 2R
(planar2r). Base dimension is determined numerically (the symbolic coefficient-rank
overcounts for n>=3); numeric vertical=8 / horizontal=7 were measured in S2-0 and
match the 2n/(2n-1) pattern that also reproduces the 2R result (4/3).
"""

import numpy as np
import pytest

from epistemic_inertial_calibration.planar2r import (
    Planar2RModel,
    inverse_dynamics_closed_form,
)
from epistemic_inertial_calibration.planar_nr import (
    PlanarNRModel,
    expected_base_dim_pattern,
    forward_kinematics,
    inverse_dynamics,
    jacobian,
    nullspace_projector,
    numeric_base_dim,
    regressor,
)


def test_nr_n2_matches_planar2r_closed_form():
    """The general n-link regressor reproduces the verified 2R closed form."""
    m2 = PlanarNRModel(n_links=2, link_lengths=(1.0, 0.9), masses=(1.2, 0.9),
                       com_dists=(0.45, 0.40), inertias=(0.07, 0.05), g=9.81)
    ref = Planar2RModel(l1=1.0, g=9.81, m1=1.2, r1=0.45, I1=0.07, m2=0.9, r2=0.40, I2=0.05)
    rng = np.random.default_rng(0)
    max_rel = 0.0
    for _ in range(300):
        q, qd, qdd = rng.uniform(-np.pi, np.pi, 2), rng.uniform(-2, 2, 2), rng.uniform(-4, 4, 2)
        a = inverse_dynamics(q, qd, qdd, m2)
        b = inverse_dynamics_closed_form(q, qd, qdd, ref)
        max_rel = max(max_rel, np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))
    assert max_rel < 1e-9


def test_nr_regressor_linear_in_beta():
    model = PlanarNRModel()
    rng = np.random.default_rng(1)
    q, qd, qdd = rng.uniform(-np.pi, np.pi, 4), rng.uniform(-2, 2, 4), rng.uniform(-4, 4, 4)
    Y = regressor(q, qd, qdd, model)
    assert Y.shape == (4, 12)
    b1, b2 = rng.normal(size=12), rng.normal(size=12)
    assert np.allclose(Y @ (b1 + b2), Y @ b1 + Y @ b2)


def test_jacobian_matches_finite_difference():
    model = PlanarNRModel()
    rng = np.random.default_rng(2)
    q = rng.uniform(-np.pi, np.pi, 4)
    J = jacobian(q, model)
    eps = 1e-6
    Jfd = np.zeros((2, 4))
    for k in range(4):
        d = np.zeros(4)
        d[k] = eps
        Jfd[:, k] = (forward_kinematics(q + d, model) - forward_kinematics(q - d, model)) / (2 * eps)
    assert np.allclose(J, Jfd, atol=1e-6)


def test_nullspace_projector():
    model = PlanarNRModel()
    rng = np.random.default_rng(3)
    q = rng.uniform(-np.pi, np.pi, 4)
    J = jacobian(q, model)
    N = nullspace_projector(q, model)
    # J N = 0 (excitation in the null-space does not move the end-effector).
    assert np.allclose(J @ N, 0.0, atol=1e-9)
    # N is an idempotent projector of rank n - 2 (4-DoF arm, 2D task).
    assert np.allclose(N @ N, N, atol=1e-9)
    assert round(float(np.trace(N))) == 2


@pytest.mark.parametrize("g,expected", [(9.81, 8), (0.0, 7)])
def test_numeric_base_dim_4r(g, expected):
    """Numeric base dim (authority): vertical=8 / horizontal=7, stable across seeds."""
    model = PlanarNRModel(g=g)
    dims = {numeric_base_dim(model, n_samples=2000, seed=s) for s in range(3)}
    assert dims == {expected}
    # Matches the 2n/(2n-1) pattern (n=4).
    assert expected == expected_base_dim_pattern(model)


@pytest.mark.parametrize("g,expected", [(9.81, 4), (0.0, 3)])
def test_numeric_base_dim_2r_consistent_with_s0(g, expected):
    """PlanarNRModel(n=2) reproduces the S0 base dimension (4/3)."""
    m2 = PlanarNRModel(n_links=2, link_lengths=(1.0, 0.9), masses=(1.2, 0.9),
                       com_dists=(0.45, 0.40), inertias=(0.07, 0.05), g=g)
    assert numeric_base_dim(m2, n_samples=2000, seed=0) == expected
