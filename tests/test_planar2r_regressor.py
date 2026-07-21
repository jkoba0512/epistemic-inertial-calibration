"""S0-3: tests for regressor linearity and correctness."""

import numpy as np
import pytest

from epistemic_inertial_calibration.planar2r import (
    BETA_LABELS,
    N_BETA,
    Planar2RModel,
    inverse_dynamics,
    inverse_dynamics_closed_form,
    regressor,
)

MODELS = {
    "vertical": Planar2RModel(g=9.81),
    "horizontal": Planar2RModel(g=0.0),
}


@pytest.mark.parametrize("name", list(MODELS))
def test_regressor_matches_closed_form(name):
    """tau = Y@beta matches the independent closed-form inverse dynamics."""
    model = MODELS[name]
    rng = np.random.default_rng(0)
    max_rel = 0.0
    for _ in range(500):
        q = rng.uniform(-np.pi, np.pi, 2)
        qd = rng.uniform(-2.0, 2.0, 2)
        qdd = rng.uniform(-4.0, 4.0, 2)
        tau_reg = inverse_dynamics(q, qd, qdd, model)
        tau_cf = inverse_dynamics_closed_form(q, qd, qdd, model)
        rel = np.max(np.abs(tau_reg - tau_cf)) / (np.max(np.abs(tau_cf)) + 1e-12)
        max_rel = max(max_rel, rel)
    assert max_rel < 1e-10, f"{name}: max rel err {max_rel:.2e}"


@pytest.mark.parametrize("name", list(MODELS))
def test_regressor_is_linear_in_beta(name):
    """Y(q,qd,qdd) does not depend on beta (linearity): Y@(b1+b2) == Y@b1 + Y@b2."""
    model = MODELS[name]
    rng = np.random.default_rng(1)
    q = rng.uniform(-np.pi, np.pi, 2)
    qd = rng.uniform(-2.0, 2.0, 2)
    qdd = rng.uniform(-4.0, 4.0, 2)
    Y = regressor(q, qd, qdd, model)
    b1 = rng.normal(size=N_BETA)
    b2 = rng.normal(size=N_BETA)
    assert np.allclose(Y @ (b1 + b2), Y @ b1 + Y @ b2)


def test_m1_column_is_structurally_zero():
    """The m1 column is identically zero (structural unidentifiability)."""
    model = Planar2RModel(g=9.81)
    rng = np.random.default_rng(2)
    m1_idx = BETA_LABELS.index("m1")
    for _ in range(50):
        q = rng.uniform(-np.pi, np.pi, 2)
        qd = rng.uniform(-2.0, 2.0, 2)
        qdd = rng.uniform(-4.0, 4.0, 2)
        Y = regressor(q, qd, qdd, model)
        assert np.allclose(Y[:, m1_idx], 0.0)
