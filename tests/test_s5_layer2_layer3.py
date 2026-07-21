"""S5-3/S5-4: tests for the 4R dynamic Layer-2 pipeline and extended Layer-3."""

from pathlib import Path

import numpy as np
import pytest

from epistemic_inertial_calibration.planar_nr import PlanarNRModel, inverse_dynamics
from epistemic_inertial_calibration.s5_layer2 import (
    ExecutionConfigNR,
    S5Layer2Settings,
    default_friction_nr,
    execute_reference_nr,
    forward_dynamics_nr,
    nominal_beta_ext_nr,
    regressor_ext_nr,
    run_layer2_closed_loop,
    stacked_regressor_ext_nr,
)
from epistemic_inertial_calibration.s5_layer3 import (
    S5ControllerConfig,
    closed_loop_rollout_ext,
    feasibility_components,
    make_beta_hat_ext,
    null_basis,
    task_failure_ext,
)
from epistemic_inertial_calibration.s5_model import (
    FrictionModel2R,
    load_s5_projection,
)
from epistemic_inertial_calibration.tasks import Task

S5_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "s5_base_floor"

pytestmark = pytest.mark.skipif(
    not (S5_DIR / "base_projection_ext_vertical.npz").exists(),
    reason="S5 base-projection artifacts not generated; run scripts/run_s5_base_floor.py",
)

FM4 = default_friction_nr(PlanarNRModel())


# --- 4R friction dynamics ---------------------------------------------------


def test_regressor_ext_nr_matches_id_plus_friction():
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = rng.uniform(-np.pi, np.pi, 4)
        qd = rng.uniform(-2, 2, 4)
        qdd = rng.uniform(-4, 4, 4)
        tau_reg = regressor_ext_nr(q, qd, qdd, FM4) @ FM4.beta_ext_true()
        tau_ref = inverse_dynamics(q, qd, qdd, FM4.rigid) + FM4.friction_torque(qd)
        np.testing.assert_allclose(tau_reg, tau_ref, atol=1e-9)


def test_forward_dynamics_nr_inverts_inverse_dynamics():
    rng = np.random.default_rng(1)
    for _ in range(5):
        q = rng.uniform(-np.pi, np.pi, 4)
        qd = rng.uniform(-2, 2, 4)
        qdd = rng.uniform(-4, 4, 4)
        tau = inverse_dynamics(q, qd, qdd, FM4.rigid) + FM4.friction_torque(qd)
        qdd_out = forward_dynamics_nr(q, qd, tau, FM4)
        np.testing.assert_allclose(qdd_out, qdd, atol=1e-8)


def test_forward_dynamics_nr_matches_2r_closed_form():
    """For n=2 the assembled forward dynamics must match the verified 2R plant."""
    from epistemic_inertial_calibration.s5_model import forward_dynamics_ext

    m2r = FrictionModel2R()
    rigid2 = PlanarNRModel(
        n_links=2, link_lengths=(1.0, 1.0),
        masses=(m2r.rigid.m1, m2r.rigid.m2),
        com_dists=(m2r.rigid.r1, m2r.rigid.r2),
        inertias=(m2r.rigid.I1, m2r.rigid.I2),
        g=m2r.rigid.g,
    )
    from epistemic_inertial_calibration.s5_layer2 import FrictionModelNR
    fm2 = FrictionModelNR(rigid=rigid2, fv=m2r.fv, fc=m2r.fc, eps=m2r.eps)
    rng = np.random.default_rng(2)
    for _ in range(5):
        q = rng.uniform(-np.pi, np.pi, 2)
        qd = rng.uniform(-2, 2, 2)
        tau = rng.uniform(-10, 10, 2)
        np.testing.assert_allclose(
            forward_dynamics_nr(q, qd, tau, fm2),
            forward_dynamics_ext(q, qd, tau, m2r),
            atol=1e-9,
        )


def test_extended_4r_rank():
    rng = np.random.default_rng(0)
    states = np.hstack([
        rng.uniform(-np.pi, np.pi, (600, 4)),
        rng.uniform(-2, 2, (600, 4)),
        rng.uniform(-4, 4, (600, 4)),
    ])
    W = stacked_regressor_ext_nr(states, FM4)
    sv = np.linalg.svd(W, compute_uv=False)
    rank = int(np.sum(sv > sv[0] * 1e-8))
    assert rank == 8 + 8  # rigid vertical 8 + 2 friction params per joint


def test_layer2_execution_tracks_task():
    task = Task(kind="hold", n_steps=80, dt=2e-3)
    settings = S5Layer2Settings(holdout_n=200)
    beta_ctrl = nominal_beta_ext_nr(FM4, settings, np.random.default_rng([0, 5]))
    q0 = np.asarray(task.q_init, float)
    ref = (np.tile(q0, (80, 1)), np.zeros((80, 4)), np.zeros((80, 4)))
    execu = execute_reference_nr(FM4, *ref, beta_ctrl, ExecutionConfigNR(),
                                 np.random.default_rng(1))
    assert not execu["fault"]
    assert execu["tracking_rmse"] < 0.05


def test_layer2_smoke_and_ordering():
    rng = np.random.default_rng(0)
    states = np.hstack([
        rng.uniform(-np.pi, np.pi, (800, 4)),
        rng.uniform(-2, 2, (800, 4)),
        rng.uniform(-4, 4, (800, 4)),
    ])
    W = stacked_regressor_ext_nr(states, FM4)
    _, sv, Vt = np.linalg.svd(W, full_matrices=False)
    rank = int(np.sum(sv > sv[0] * 1e-8))
    V_base = Vt[:rank, :]
    task = Task(kind="hold", n_steps=250, dt=2e-3)
    res = run_layer2_closed_loop(
        FM4, V_base, task, S5Layer2Settings(holdout_n=200), ExecutionConfigNR(),
        n_seeds=1,
    )
    s = res["summary"]
    assert s["no_exploration"]["n_faults"] == 0
    # Null-space excitation must identify better than pure task motion.
    assert (s["fourier_nullspace"]["logdet_cov"]["mean"]
            < s["no_exploration"]["logdet_cov"]["mean"])
    # The task must stay held under excitation (executed motion, not assumed).
    assert s["fourier_nullspace"]["ee_error_max"]["mean"] < 0.08


# --- extended Layer-3 -------------------------------------------------------


def _proj2r():
    return load_s5_projection(S5_DIR / "base_projection_ext_vertical.npz")


def test_feasibility_components_velocity_blindness():
    """Short-horizon far target: velocity component must dominate the torque one."""
    proj = _proj2r()
    fmodel = proj.model()
    beta = fmodel.beta_ext_true()
    cfg_slow = S5ControllerConfig(n_steps=750)
    cfg_fast = S5ControllerConfig(n_steps=140)
    near, far = (0.9, -0.6), (1.6, -1.3)
    f_slow = feasibility_components(fmodel, beta, (0.2, 0.3), near, cfg_slow)
    f_fast = feasibility_components(fmodel, beta, (0.2, 0.3), far, cfg_fast)
    assert f_slow["rho_vel"] < 1.0
    assert f_fast["rho_vel"] > 1.0  # desired velocity exceeds the actuator rating
    assert f_fast["rho"] >= f_fast["rho_vel"]


def test_velocity_fault_counts_as_failure():
    proj = _proj2r()
    fmodel = proj.model()
    beta = fmodel.beta_ext_true()
    # Aggressive short-horizon far target with loose torque: the plant CAN
    # reach high velocity, so the velocity fault (not torque) must trigger.
    cfg = S5ControllerConfig(torque_limit=60.0, n_steps=140)
    r = closed_loop_rollout_ext(fmodel, beta, (0.2, 0.3), (1.6, -1.3), cfg)
    assert r["fault"]
    assert task_failure_ext(r, cfg)


def test_pin_shift_invisible_to_calibration_metric():
    proj = _proj2r()
    fmodel = proj.model()
    V = proj.V_base
    Nb = null_basis(V)
    beta_true = fmodel.beta_ext_true()
    rng = np.random.default_rng(3)
    beta_pin = make_beta_hat_ext(beta_true, V, Nb, 0.0, 0.5, rng)
    from epistemic_inertial_calibration.s5_layer3 import calibration_alpha_rmse_ext

    assert calibration_alpha_rmse_ext(beta_pin, beta_true, V) < 1e-10
    assert float(np.linalg.norm(beta_pin - beta_true)) > 0.1
