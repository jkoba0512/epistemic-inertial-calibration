"""S3-0: 2R forward dynamics, FF+PD controller, and closed-loop reach-and-hold rollout.

Layer 3 (terminal controllability). The torque mainline (S1/S2) is inverse dynamics
only; S3 needs forward simulation. The 2R M, C, g are linear in the barycentric
parameters beta = [m1, h1, J1, m2, h2, J2] and reuse the structure verified against
planar2r.inverse_dynamics_closed_form (~1e-15).

Calibration error enters through the feedforward term: the controller uses an
estimated beta_hat (the regressor depends only on link length and gravity, not on
beta), so tau_ff = Y(q,qd,qdd_des) @ beta_hat carries the parameter error into the
torque, while the true plant integrates with beta_true.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .planar2r import Planar2RModel, regressor


def mcg_2r(q, qd, beta, l1: float, g: float):
    """Mass matrix M (2x2), Coriolis vector c (2,), gravity vector gv (2,) from beta."""
    m1, h1, J1, m2, h2, J2 = beta
    q1, q2 = q
    dq1, dq2 = qd
    c2, s2 = np.cos(q2), np.sin(q2)
    M = np.array([
        [J1 + J2 + l1**2 * m2 + 2.0 * l1 * h2 * c2, J2 + l1 * h2 * c2],
        [J2 + l1 * h2 * c2, J2],
    ])
    h = l1 * h2 * s2
    c_vec = np.array([-h * (2.0 * dq1 * dq2 + dq2**2), h * dq1**2])
    gv = np.array([
        (h1 + l1 * m2) * g * np.cos(q1) + h2 * g * np.cos(q1 + q2),
        h2 * g * np.cos(q1 + q2),
    ])
    return M, c_vec, gv


def forward_dynamics_2r(q, qd, tau, beta, l1: float, g: float) -> np.ndarray:
    """qdd = M^{-1} (tau - C qd - g)."""
    M, c_vec, gv = mcg_2r(q, qd, beta, l1, g)
    return np.linalg.solve(M, np.asarray(tau, float) - c_vec - gv)


# ---------------------------------------------------------------------------
# Reference trajectory (joint-space reach then hold)
# ---------------------------------------------------------------------------


def _minjerk(u):
    return 10 * u**3 - 15 * u**4 + 6 * u**5, (30 * u**2 - 60 * u**3 + 30 * u**4)


def reach_hold_trajectory(q_start, q_goal, n_steps: int, dt: float, reach_frac: float = 0.6):
    """Min-jerk reach to q_goal over the first reach_frac of the horizon, then hold."""
    q_start = np.asarray(q_start, float)
    q_goal = np.asarray(q_goal, float)
    n = q_start.size
    q_des = np.zeros((n_steps, n))
    qd_des = np.zeros((n_steps, n))
    qdd_des = np.zeros((n_steps, n))
    reach_steps = max(1, int(reach_frac * n_steps))
    T = reach_steps * dt
    for k in range(n_steps):
        if k < reach_steps:
            u = k / reach_steps
            s, sdot = _minjerk(u)
            sdd = (60 * u - 180 * u**2 + 120 * u**3) / (T**2)
            q_des[k] = q_start + s * (q_goal - q_start)
            qd_des[k] = (sdot / T) * (q_goal - q_start)
            qdd_des[k] = sdd * (q_goal - q_start)
        else:
            q_des[k] = q_goal
    return q_des, qd_des, qdd_des


# ---------------------------------------------------------------------------
# Closed-loop rollout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControllerConfig:
    kp: float = 60.0
    kd: float = 15.0
    torque_limit: float = 40.0  # per-joint |tau| saturation
    dt: float = 2e-3
    n_steps: int = 750  # 1.5 s horizon
    settle_frac: float = 0.2  # final fraction used as the hold/settle window
    pos_threshold: float = 0.05  # rad
    vel_threshold: float = 0.05  # rad/s


def closed_loop_rollout(
    model: Planar2RModel,
    beta_hat: np.ndarray,
    q_start,
    q_goal,
    cfg: ControllerConfig,
    beta_plant: np.ndarray | None = None,
) -> dict:
    """Simulate inverse-dynamics FF + PD reach-and-hold under the true plant.

    The plant integrates with beta_plant (default model.beta_true()); the controller
    uses beta_hat. Setting beta_plant=beta_hat gives the controller's self-consistent
    rollout (used for the model-based rollout risk). Semi-implicit (symplectic) Euler.
    Torque saturated to +/- cfg.torque_limit.
    """
    beta_true = model.beta_true() if beta_plant is None else np.asarray(beta_plant, float)
    l1, g = model.l1, model.g
    q = np.asarray(q_start, float).copy()
    qd = np.zeros(2)
    q_des, qd_des, qdd_des = reach_hold_trajectory(q_start, q_goal, cfg.n_steps, cfg.dt)

    qs = np.zeros((cfg.n_steps, 2))
    qds = np.zeros((cfg.n_steps, 2))
    sat_hits = 0
    diverged = False
    diverge_bound = 1e3  # rad or rad/s; beyond this the controlled system has blown up
    for k in range(cfg.n_steps):
        tau_ff = regressor(q, qd, qdd_des[k], model) @ beta_hat
        tau = tau_ff + cfg.kp * (q_des[k] - q) + cfg.kd * (qd_des[k] - qd)
        tau_sat = np.clip(tau, -cfg.torque_limit, cfg.torque_limit)
        sat_hits += int(np.sum(np.abs(tau - tau_sat) > 1e-9))
        qdd = forward_dynamics_2r(q, qd, tau_sat, beta_true, l1, g)
        qd = qd + qdd * cfg.dt
        q = q + qd * cfg.dt
        qs[k] = q
        qds[k] = qd
        if not np.all(np.isfinite(q)) or np.max(np.abs(q)) > diverge_bound \
                or np.max(np.abs(qd)) > diverge_bound:
            # Controller could not stabilize; mark the rest as diverged (a failure).
            diverged = True
            qs[k:] = q
            qds[k:] = qd
            break

    settle = max(1, int(cfg.settle_frac * cfg.n_steps))
    q_goal = np.asarray(q_goal, float)
    finite = bool(not diverged and np.all(np.isfinite(qs)) and np.all(np.isfinite(qds)))
    if finite:
        hold_err = float(np.mean(np.linalg.norm(qs[-settle:] - q_goal, axis=1)))
        hold_vel = float(np.mean(np.linalg.norm(qds[-settle:], axis=1)))
        term_err = float(np.linalg.norm(qs[-1] - q_goal))
        term_vel = float(np.linalg.norm(qds[-1]))
    else:
        hold_err = hold_vel = term_err = term_vel = float("inf")
    return {
        "q": qs, "qd": qds,
        "terminal_joint_error": term_err,
        "terminal_velocity_residual": term_vel,
        "hold_error": hold_err,
        "hold_velocity": hold_vel,
        "saturation_fraction": sat_hits / (cfg.n_steps * 2),
        "finite": finite,
        "diverged": diverged,
    }


def task_failure(rollout: dict, cfg: ControllerConfig) -> bool:
    """Reach-and-hold failure: settle-window mean joint error or velocity over threshold."""
    if not rollout["finite"]:
        return True
    return bool(rollout["hold_error"] > cfg.pos_threshold or rollout["hold_velocity"] > cfg.vel_threshold)
