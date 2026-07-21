"""S5-5: tests for the iiwa hardware-executable pipeline (requires Pinocchio)."""

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")

from epistemic_inertial_calibration.s5_iiwa import (  # noqa: E402
    ExecutionConfigIiwa,
    IiwaL3Config,
    Q_START,
    build_observations_iiwa,
    execute_reference_iiwa,
    feasibility_components_iiwa,
    iiwa_rollout_ext,
    load_friction_iiwa,
    nominal_beta_ext_iiwa,
    regressor_ext_iiwa,
    sample_states_hw,
    stacked_regressor_ext_iiwa,
    _free_reference_iiwa,
    _nullspace_reference,
)

FM = load_friction_iiwa()


def test_extended_regressor_consistency():
    """Y_ext @ beta_ext == rnea + armature torque + friction torque."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        s = sample_states_hw(FM, 1, seed=int(rng.integers(1e6)))[0]
        q, qd, qdd = s[:7], s[7:14], s[14:21]
        tau_reg = regressor_ext_iiwa(q, qd, qdd, FM) @ FM.beta_ext_true()
        tau_ref = (FM.base.inverse_dynamics(q, qd, qdd)
                   + np.asarray(FM.armature) * qdd + FM.friction_torque(qd))
        np.testing.assert_allclose(tau_reg, tau_ref, atol=1e-8)


def test_forward_dynamics_inverts():
    s = sample_states_hw(FM, 1, seed=3)[0]
    q, qd, qdd = s[:7], s[7:14], s[14:21]
    tau = (FM.base.inverse_dynamics(q, qd, qdd)
           + np.asarray(FM.armature) * qdd + FM.friction_torque(qd))
    np.testing.assert_allclose(FM.forward_dynamics(q, qd, tau), qdd, atol=1e-7)


def test_extended_base_rank_within_hw_limits():
    W = stacked_regressor_ext_iiwa(sample_states_hw(FM, 400, seed=0), FM)
    sv = np.linalg.svd(W, compute_uv=False)
    rank = int(np.sum(sv > sv[0] * 1e-8))
    # rigid numeric base 43 + 14 friction + 5 of 7 armature columns (two are
    # linearly dependent with rigid inertia directions and get absorbed).
    assert rank == 62


def test_free_reference_respects_limits_and_executes():
    cfg = ExecutionConfigIiwa()
    beta_ctrl = nominal_beta_ext_iiwa(FM, np.random.default_rng([0, 5]))
    ref = _free_reference_iiwa(FM, 1000, cfg.dt, Q_START, beta_ctrl, cfg,
                               np.random.default_rng(1))
    vlim = np.array(FM.base.model.velocityLimit, float)
    assert np.all(np.abs(ref[1]) <= vlim[None, :] + 1e-9)
    execu = execute_reference_iiwa(FM, *ref, beta_ctrl, cfg, np.random.default_rng(2))
    assert not execu["fault"]
    assert execu["tracking_rmse"] < 0.05
    # no sample-rate buzz
    qdd_step = np.diff(execu["qd_true"], axis=0) / cfg.dt
    assert float(np.sqrt(np.mean(qdd_step**2))) < 20.0


def test_nullspace_reference_holds_ee_when_executed():
    cfg = ExecutionConfigIiwa()
    beta_ctrl = nominal_beta_ext_iiwa(FM, np.random.default_rng([0, 5]))
    ref = _nullspace_reference(FM, Q_START, 800, cfg.dt, amp=0.4, seed=0)
    execu = execute_reference_iiwa(FM, *ref, beta_ctrl, cfg, np.random.default_rng(3))
    assert not execu["fault"]
    p0 = FM.base.forward_kinematics(np.asarray(Q_START, float))
    ee = max(
        float(np.linalg.norm(FM.base.forward_kinematics(qk) - p0))
        for qk in execu["q_true"][::50]
    )
    assert ee < 0.05  # executed (not assumed) task hold


def test_observations_consistent_with_truth():
    cfg = ExecutionConfigIiwa()
    beta_ctrl = nominal_beta_ext_iiwa(FM, np.random.default_rng([0, 5]))
    ref = _free_reference_iiwa(FM, 1000, cfg.dt, Q_START, beta_ctrl, cfg,
                               np.random.default_rng(1))
    execu = execute_reference_iiwa(FM, *ref, beta_ctrl, cfg, np.random.default_rng(2))
    W, y = build_observations_iiwa(execu, np.eye(91), FM, cfg)
    assert y.size > 0
    resid = y - W @ FM.beta_ext_true()
    assert float(np.sqrt(np.mean(resid**2))) < 2.0


def test_velocity_only_infeasible_condition_faults():
    """far/normal exceeds the velocity rating with torque to spare (measured
    earlier: peak qd 3.12 vs limit 1.48) -> rho_vel flags it, tau-only misses,
    and the executed rollout faults."""
    cfg = IiwaL3Config(torque_scale=1.0, n_steps=1500)
    from epistemic_inertial_calibration.s5_iiwa import TARGETS
    f = feasibility_components_iiwa(FM, FM.beta_ext_true(), Q_START, TARGETS["far"], cfg)
    assert f["rho_vel"] > 1.0
    r = iiwa_rollout_ext(FM, FM.beta_ext_true(), Q_START, TARGETS["far"], cfg)
    assert r["fault"]


def test_feasible_condition_succeeds():
    cfg = IiwaL3Config(torque_scale=1.0, n_steps=1500)
    from epistemic_inertial_calibration.s5_iiwa import TARGETS, iiwa_failure_ext
    f = feasibility_components_iiwa(FM, FM.beta_ext_true(), Q_START, TARGETS["near"], cfg)
    assert f["rho"] < 1.0
    r = iiwa_rollout_ext(FM, FM.beta_ext_true(), Q_START, TARGETS["near"], cfg)
    assert not iiwa_failure_ext(r, cfg)
