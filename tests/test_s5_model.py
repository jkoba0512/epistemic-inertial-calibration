"""S5-0: tests for the friction-extended 2R model."""

import numpy as np
import pytest

from epistemic_inertial_calibration.controllers import forward_dynamics_2r
from epistemic_inertial_calibration.planar2r import (
    Planar2RModel,
    inverse_dynamics_closed_form,
)
from epistemic_inertial_calibration.s5_model import (
    BETA_EXT_LABELS,
    N_BETA_EXT,
    FrictionModel2R,
    forward_dynamics_ext,
    inverse_dynamics_ext,
    load_s5_projection,
    regressor_ext,
    save_s5_projection,
    stacked_regressor_ext,
)

FM = FrictionModel2R()
RNG = np.random.default_rng(0)


def _random_state(rng):
    q = rng.uniform(-np.pi, np.pi, 2)
    qd = rng.uniform(-2.0, 2.0, 2)
    qdd = rng.uniform(-4.0, 4.0, 2)
    return q, qd, qdd


def test_labels_and_shape():
    assert N_BETA_EXT == 10
    assert BETA_EXT_LABELS[:6] == ("m1", "h1", "J1", "m2", "h2", "J2")
    assert BETA_EXT_LABELS[6:] == ("Fv1", "Fc1", "Fv2", "Fc2")
    q, qd, qdd = _random_state(np.random.default_rng(1))
    assert regressor_ext(q, qd, qdd, FM).shape == (2, 10)


def test_extended_regressor_matches_closed_form_plus_friction():
    """Y_ext @ beta_ext == closed-form rigid torque + friction torque (~1e-12)."""
    rng = np.random.default_rng(2)
    for _ in range(20):
        q, qd, qdd = _random_state(rng)
        tau_reg = inverse_dynamics_ext(q, qd, qdd, FM)
        tau_ref = inverse_dynamics_closed_form(q, qd, qdd, FM.rigid) + FM.friction_torque(qd)
        np.testing.assert_allclose(tau_reg, tau_ref, atol=1e-10)


def test_friction_vanishes_at_rest():
    q = np.array([0.3, -0.7])
    zero = np.zeros(2)
    tau = inverse_dynamics_ext(q, zero, zero, FM)
    tau_rigid = inverse_dynamics_closed_form(q, zero, zero, FM.rigid)
    np.testing.assert_allclose(tau, tau_rigid, atol=1e-12)


def test_forward_dynamics_inverts_inverse_dynamics():
    """qdd_out of the friction plant under tau = ID_ext(qdd_in) recovers qdd_in."""
    rng = np.random.default_rng(3)
    for _ in range(10):
        q, qd, qdd = _random_state(rng)
        tau = inverse_dynamics_ext(q, qd, qdd, FM)
        qdd_out = forward_dynamics_ext(q, qd, tau, FM)
        np.testing.assert_allclose(qdd_out, qdd, atol=1e-9)


def test_friction_dissipates_against_motion():
    """Friction torque opposes velocity: plant decelerates faster than rigid."""
    q = np.array([0.2, 0.4])
    qd = np.array([1.0, -1.0])
    tau = np.zeros(2)
    qdd_fric = forward_dynamics_ext(q, qd, tau, FM)
    qdd_rigid = forward_dynamics_2r(q, qd, tau, FM.rigid.beta_true(), FM.l1, FM.g)
    # Friction adds torque opposing qd; M^{-1} is positive definite, so the
    # velocity-aligned component of the acceleration difference is negative.
    diff = qdd_fric - qdd_rigid
    assert float(diff @ qd) < 0.0


def test_extended_rank_vertical_and_horizontal():
    from epistemic_inertial_calibration.base_parameters import SamplingConfig, sample_states

    for g, expected in ((9.81, 8), (0.0, 7)):
        fm = FrictionModel2R(rigid=Planar2RModel(g=g))
        states = sample_states(SamplingConfig(regime="dynamic", n_samples=400, seed=0))
        W = stacked_regressor_ext(states, fm)
        sv = np.linalg.svd(W, compute_uv=False)
        rank = int(np.sum(sv > sv[0] * 1e-8))
        assert rank == expected, (g, rank)


def test_projection_round_trip(tmp_path):
    from epistemic_inertial_calibration.base_parameters import SamplingConfig, sample_states

    states = sample_states(SamplingConfig(regime="dynamic", n_samples=300, seed=0))
    W = stacked_regressor_ext(states, FM)
    _, sv, Vt = np.linalg.svd(W, full_matrices=False)
    rank = int(np.sum(sv > sv[0] * 1e-8))
    path = tmp_path / "proj.npz"
    save_s5_projection(path, Vt[:rank], rank, FM, 1e-8)
    proj = load_s5_projection(path)
    assert proj.rank == rank
    assert proj.beta_labels == BETA_EXT_LABELS
    np.testing.assert_allclose(proj.V_base, Vt[:rank])
    fm2 = proj.model()
    assert fm2.fv == FM.fv and fm2.fc == FM.fc and fm2.eps == FM.eps
    assert fm2.g == FM.g


def test_regressor_null_direction_is_torque_invisible():
    """m1 stays structurally unidentifiable in the extended model."""
    rng = np.random.default_rng(4)
    states = np.hstack([
        rng.uniform(-np.pi, np.pi, (100, 2)),
        rng.uniform(-2, 2, (100, 2)),
        rng.uniform(-4, 4, (100, 2)),
    ])
    W = stacked_regressor_ext(states, FM)
    e_m1 = np.zeros(10)
    e_m1[0] = 1.0
    assert np.max(np.abs(W @ e_m1)) == pytest.approx(0.0, abs=1e-12)
