"""S5-2: closed-loop reference execution and hardware-realistic observation (2R).

The designed reference (s5_trajectories) is tracked by an inverse-dynamics
feedforward + PD controller that only knows the NOMINAL parameters; the plant
integrates the true friction dynamics. What the identification pipeline sees
is exactly what a hardware logger would provide:

  q_meas   = q_true  + encoder noise
  qd_meas  = qd_true + velocity-estimate noise
  tau_meas = applied (saturated) torque + torque-sensor noise

Accelerations are NOT measured: qdd is reconstructed offline from qd_meas by
zero-phase smoothing + central differencing, the standard batch-IDIM
practice. The identification regressor is built from the measured/estimated
states only. True states are recorded separately for the evaluator and are
never given to the estimator.

A safety monitor checks the true state against fault bounds every step
(position and velocity, the way an industrial controller faults out) and
aborts the rollout with a fault flag if violated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .s5_model import (
    FrictionModel2R,
    S5ProjectionArtifact,
    forward_dynamics_ext,
    regressor_ext,
    stacked_regressor_ext,
)


@dataclass(frozen=True)
class ExecutionConfig:
    """Controller, sensor, and safety settings for closed-loop execution."""

    kp: float = 60.0
    kd: float = 15.0
    torque_limit: float = 40.0
    dt: float = 2e-3
    sigma_q: float = 1e-4       # encoder noise (rad)
    sigma_qd: float = 1e-3      # velocity-estimate noise (rad/s)
    sigma_tau: float = 1e-2     # torque-sensor noise (N m)
    q_fault: float = 1.15 * np.pi   # position fault bound (rad)
    qd_fault: float = 2.5           # velocity fault bound (rad/s)
    smooth_window: int = 11     # zero-phase smoothing window for qdd estimation (odd)
    obs_decimate: int = 5       # keep every k-th sample as an observation
    edge_trim: int = 15         # drop this many samples at both ends (filter edges)
    qd_min: float = 0.05        # drop joint-rows with |qd_meas| below this (rad/s):
    #                             near rest the differentiated noise dominates the
    #                             inertia/friction columns and biases the estimate
    #                             (standard batch-IDIM velocity filtering)


def execute_reference(
    fmodel: FrictionModel2R,
    q_des: np.ndarray,
    qd_des: np.ndarray,
    qdd_des: np.ndarray,
    beta_ctrl: np.ndarray,
    cfg: ExecutionConfig,
    rng: np.random.Generator,
    q_init: np.ndarray | None = None,
    qd_init: np.ndarray | None = None,
) -> dict:
    """Track the reference on the true friction plant; return logged data.

    beta_ctrl is the extended nominal parameter vector the controller uses for
    the feedforward term (true parameters are unknown to the controller).
    """
    n_steps = q_des.shape[0]
    q = (q_des[0] if q_init is None else np.asarray(q_init, float)).copy()
    qd = (np.zeros(2) if qd_init is None else np.asarray(qd_init, float)).copy()

    q_true = np.zeros((n_steps, 2))
    qd_true = np.zeros((n_steps, 2))
    q_meas = np.zeros((n_steps, 2))
    qd_meas = np.zeros((n_steps, 2))
    tau_meas = np.zeros((n_steps, 2))
    sat_hits = 0
    fault = False
    fault_step = -1

    for k in range(n_steps):
        qm = q + rng.normal(0.0, cfg.sigma_q, 2)
        qdm = qd + rng.normal(0.0, cfg.sigma_qd, 2)
        # FF from the nominal model at the measured state and desired accel.
        tau_ff = regressor_ext(qm, qdm, qdd_des[k], fmodel) @ beta_ctrl
        tau = tau_ff + cfg.kp * (q_des[k] - qm) + cfg.kd * (qd_des[k] - qdm)
        tau_sat = np.clip(tau, -cfg.torque_limit, cfg.torque_limit)
        sat_hits += int(np.sum(np.abs(tau - tau_sat) > 1e-9))

        qdd = forward_dynamics_ext(q, qd, tau_sat, fmodel)
        qd = qd + qdd * cfg.dt
        q = q + qd * cfg.dt

        q_true[k] = q
        qd_true[k] = qd
        q_meas[k] = qm
        qd_meas[k] = qdm
        tau_meas[k] = tau_sat + rng.normal(0.0, cfg.sigma_tau, 2)

        if np.max(np.abs(q)) > cfg.q_fault or np.max(np.abs(qd)) > cfg.qd_fault:
            fault = True
            fault_step = k
            n_steps = k + 1  # truncate the log at the fault
            q_true = q_true[:n_steps]
            qd_true = qd_true[:n_steps]
            q_meas = q_meas[:n_steps]
            qd_meas = qd_meas[:n_steps]
            tau_meas = tau_meas[:n_steps]
            q_des = q_des[:n_steps]
            break

    track_err = q_true - q_des[: q_true.shape[0]]
    return {
        "q_true": q_true, "qd_true": qd_true,
        "q_meas": q_meas, "qd_meas": qd_meas, "tau_meas": tau_meas,
        "fault": fault, "fault_step": fault_step,
        "saturation_fraction": sat_hits / max(1, q_true.shape[0] * 2),
        "tracking_rmse": float(np.sqrt(np.mean(track_err**2))),
        "n_steps": q_true.shape[0],
    }


def estimate_qdd(qd_meas: np.ndarray, dt: float, smooth_window: int) -> np.ndarray:
    """Offline acceleration estimate: zero-phase smoothing + central difference.

    Symmetric moving-average smoothing (applied forward once with a centered
    kernel, hence zero phase) of the measured velocity, then np.gradient.
    Batch (acausal) processing is the standard offline-IDIM practice; the
    filter edges are handled by the caller's edge trim.
    """
    w = max(1, smooth_window | 1)  # force odd
    kernel = np.ones(w) / w
    pad = w // 2
    sm = np.empty_like(qd_meas)
    for j in range(qd_meas.shape[1]):
        padded = np.pad(qd_meas[:, j], pad, mode="edge")
        sm[:, j] = np.convolve(padded, kernel, mode="valid")
    return np.gradient(sm, dt, axis=0)


def build_observations(
    execu: dict,
    proj: S5ProjectionArtifact,
    fmodel: FrictionModel2R,
    cfg: ExecutionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measured data -> (states_meas, W_base, y) for the estimator.

    states_meas stacks [q_meas | qd_meas | qdd_est]; y interleaves the two
    joints per kept sample, matching the stacked-regressor row order.
    Joint-rows with |qd_meas| < qd_min are dropped (velocity filtering): a
    near-rest joint contributes only differentiated noise to the inertia and
    friction columns, which confidently biases the estimate.
    """
    qdd_est = estimate_qdd(execu["qd_meas"], cfg.dt, cfg.smooth_window)
    states = np.hstack([execu["q_meas"], execu["qd_meas"], qdd_est])
    tau = execu["tau_meas"]
    t0, t1 = cfg.edge_trim, states.shape[0] - cfg.edge_trim
    if t1 - t0 < cfg.obs_decimate:
        raise ValueError("rollout too short after edge trim")
    keep = np.arange(t0, t1, cfg.obs_decimate)
    states = states[keep]
    y = tau[keep].reshape(-1)
    W_base = stacked_regressor_ext(states, fmodel) @ proj.V_base.T
    row_qd = np.abs(states[:, 2:4]).reshape(-1)  # |qd_meas| per (sample, joint) row
    mask = row_qd >= cfg.qd_min
    return states, W_base[mask], y[mask]
