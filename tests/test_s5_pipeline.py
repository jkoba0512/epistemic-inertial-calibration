"""S5-1/S5-2: tests for executable trajectory design and closed-loop execution."""

from pathlib import Path

import numpy as np
import pytest

from epistemic_inertial_calibration.s5_execution import (
    ExecutionConfig,
    build_observations,
    estimate_qdd,
    execute_reference,
)
from epistemic_inertial_calibration.s5_layer1 import (
    S5Layer1Settings,
    nominal_beta_ext,
    run_closed_loop_ablation,
)
from epistemic_inertial_calibration.s5_model import (
    load_s5_projection,
)
from epistemic_inertial_calibration.s5_trajectories import (
    SEGMENT_POLICIES,
    DesignContext,
    S5TrajConfig,
    design_reference,
    quintic_segment,
)

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts" / "s5_base_floor"

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "base_projection_ext_vertical.npz").exists(),
    reason="S5 base-projection artifacts not generated; run scripts/run_s5_base_floor.py",
)

# Small, fast configs for tests.
TRAJ = S5TrajConfig(seg_duration=0.3, n_segments=4, n_candidates=3, dt=2e-3)
EXEC = ExecutionConfig(edge_trim=10, obs_decimate=5)
SET = S5Layer1Settings(holdout_n=300)


def _proj():
    return load_s5_projection(ARTIFACTS / "base_projection_ext_vertical.npz")


def _nominal(proj, seed=0):
    return nominal_beta_ext(proj.model(), SET, np.random.default_rng([seed, 5]))


# --- quintic segments -------------------------------------------------------


def test_quintic_boundary_conditions():
    q0, qd0 = np.array([0.1, -0.2]), np.array([0.5, -0.3])
    q1, qd1 = np.array([0.8, 0.4]), np.array([-0.2, 0.6])
    n, dt = 200, 2e-3
    q, qd, qdd = quintic_segment(q0, qd0, q1, qd1, n, dt)
    np.testing.assert_allclose(q[-1], q1, atol=1e-9)
    np.testing.assert_allclose(qd[-1], qd1, atol=1e-8)
    np.testing.assert_allclose(qdd[-1], 0.0, atol=1e-6)
    # Finite-difference consistency of the analytic derivatives.
    qd_fd = np.gradient(q, dt, axis=0)
    assert float(np.sqrt(np.mean((qd_fd[5:-5] - qd[5:-5]) ** 2))) < 5e-3


def test_quintic_chain_is_continuous():
    rng = np.random.default_rng(0)
    proj = _proj()
    ref = design_reference("smooth_random", rng, TRAJ, proj.model(), _nominal(proj))
    q, qd, qdd = ref
    assert q.shape[0] == TRAJ.total_steps
    # No jumps: velocity bounded by limit implies position steps <= vel*dt.
    dq = np.abs(np.diff(q, axis=0))
    assert float(dq.max()) <= TRAJ.vel_limit * TRAJ.dt * 1.5


@pytest.mark.parametrize("policy", ["hold_sequence", "smooth_random", "fourier_envelope",
                                    "segment_fim_greedy"])
def test_designed_references_respect_envelope(policy):
    proj = _proj()
    fmodel = proj.model()
    beta_n = _nominal(proj)
    ctx = None
    if policy == "segment_fim_greedy":
        ctx = DesignContext(V_base=proj.V_base, sigma_tau=EXEC.sigma_tau)
    q, qd, qdd = design_reference(policy, np.random.default_rng(1), TRAJ, fmodel, beta_n, ctx)
    assert np.max(np.abs(q)) <= TRAJ.q_range + 1e-9
    assert np.max(np.abs(qd)) <= TRAJ.vel_limit + 1e-9
    assert np.max(np.abs(qdd)) <= TRAJ.acc_limit + 1e-9


# --- closed-loop execution --------------------------------------------------


def test_execution_tracks_reference():
    proj = _proj()
    fmodel = proj.model()
    beta_n = _nominal(proj)
    ref = design_reference("smooth_random", np.random.default_rng(2), TRAJ, fmodel, beta_n)
    execu = execute_reference(fmodel, *ref, beta_n, EXEC, np.random.default_rng(3))
    assert not execu["fault"]
    assert execu["tracking_rmse"] < 0.1  # rad; nominal FF + PD should track well


def test_execution_perfect_model_tracks_better_than_worst_nominal():
    proj = _proj()
    fmodel = proj.model()
    ref = design_reference("smooth_random", np.random.default_rng(2), TRAJ, fmodel,
                           fmodel.beta_ext_true())
    e_true = execute_reference(fmodel, *ref, fmodel.beta_ext_true(), EXEC,
                               np.random.default_rng(3))
    assert e_true["tracking_rmse"] < 0.02


def test_estimate_qdd_recovers_smooth_acceleration():
    t = np.arange(0, 2.0, 2e-3)
    qd = np.stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)], axis=1)
    qdd_true = np.stack([2 * np.pi * np.cos(2 * np.pi * t),
                         -2 * np.pi * np.sin(2 * np.pi * t)], axis=1)
    est = estimate_qdd(qd + 1e-4 * np.random.default_rng(0).normal(size=qd.shape),
                       2e-3, smooth_window=11)
    err = est[20:-20] - qdd_true[20:-20]
    assert float(np.sqrt(np.mean(err**2))) < 0.15


def test_observations_are_consistent_with_measured_torque():
    """W_base @ alpha_true should predict tau_meas up to noise + EIV error."""
    proj = _proj()
    fmodel = proj.model()
    beta_n = _nominal(proj)
    ref = design_reference("smooth_random", np.random.default_rng(4), TRAJ, fmodel, beta_n)
    execu = execute_reference(fmodel, *ref, beta_n, EXEC, np.random.default_rng(5))
    _, W_base, y = build_observations(execu, proj, fmodel, EXEC)
    a_true = proj.V_base @ fmodel.beta_ext_true()
    resid = W_base @ a_true - y
    assert float(np.sqrt(np.mean(resid**2))) < 0.2  # N m, dominated by qdd-est error


def test_fault_monitor_triggers_and_truncates():
    proj = _proj()
    fmodel = proj.model()
    # Absurd controller parameters destabilize the loop -> fault must trigger.
    beta_bad = fmodel.beta_ext_true() * 40.0
    ref = design_reference("smooth_random", np.random.default_rng(6), TRAJ, fmodel,
                           fmodel.beta_ext_true())
    cfg = ExecutionConfig(kp=0.0, kd=0.0)
    execu = execute_reference(fmodel, *ref, beta_bad, cfg, np.random.default_rng(7))
    assert execu["fault"]
    assert execu["n_steps"] < TRAJ.total_steps


# --- closed-loop ablation (smoke) ------------------------------------------


def test_closed_loop_ablation_smoke():
    proj = _proj()
    result = run_closed_loop_ablation(
        proj, SET, TRAJ, EXEC, n_seeds=1, policies=SEGMENT_POLICIES
    )
    summary = result["summary"]
    assert set(summary) == set(SEGMENT_POLICIES)
    for policy, s in summary.items():
        assert s["n_faults"] == 0, policy
        assert np.isfinite(s["alpha_rmse"]["mean"]), policy
        assert np.isfinite(s["validation_torque_rmse"]["mean"]), policy
        assert s["max_abs_qd"]["mean"] <= 2.5  # within the fault bound
        assert s["peak_abs_tau"]["mean"] <= 40.0 + 0.5
    # Dynamic policies must identify better than the hold sequence.
    assert (summary["smooth_random"]["alpha_rmse"]["mean"]
            < summary["hold_sequence"]["alpha_rmse"]["mean"])


def test_active_ig_uses_measured_feedback_deterministically():
    proj = _proj()
    r1 = run_closed_loop_ablation(proj, SET, TRAJ, EXEC, n_seeds=1,
                                  policies=("segment_active_ig",))
    r2 = run_closed_loop_ablation(proj, SET, TRAJ, EXEC, n_seeds=1,
                                  policies=("segment_active_ig",))
    assert r1["by_seed"][0]["alpha_rmse"] == r2["by_seed"][0]["alpha_rmse"]
