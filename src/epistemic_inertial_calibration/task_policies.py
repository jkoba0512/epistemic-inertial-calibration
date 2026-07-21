"""S2-A: task-compatible (null-space) excitation policies and rollouts.

Motion law (resolved-rate with null-space excitation and drift correction):
    qd = J^+ (pdot_des + kp (p_des - p_ee)) + N(q) xi
The first term tracks the task (the kp feedback cancels discrete-integration drift);
the second injects excitation xi into the null space N = I - J^+ J, which does not
move the end-effector. A rollout returns the state sequence (q|qd|qdd) for the
regressor, plus the EE task error and the null-space effort.

S2-A policies (null-space strict): no_exploration, random_nullspace, fourier_nullspace.
The active/greedy task-compatible policies come in S2-B/C.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .planar_nr import PlanarNRModel, forward_kinematics, jacobian
from .tasks import Task

NULLSPACE_POLICIES = ("no_exploration", "random_nullspace", "fourier_nullspace")


@dataclass(frozen=True)
class NullspaceConfig:
    amp: float = 1.0
    fourier_modes: int = 4
    base_freq: float = 0.5  # Hz
    rcond: float = 1e-10
    smooth_window: int = 15  # moving-average window for smooth_random (physical qdd)


def _excitation(policy: str, n: int, task: Task, cfg: NullspaceConfig, seed: int):
    """Return a function xi(k) -> (n,) joint-space excitation (pre null-space projection)."""
    if policy == "no_exploration":
        return lambda k: np.zeros(n)
    rng = np.random.default_rng(seed)
    if policy == "random_nullspace":
        raw = rng.normal(0.0, cfg.amp, size=(task.n_steps, n))
        return lambda k: raw[k]
    if policy == "smooth_random_nullspace":
        # Low-pass (moving-average) filtered iid noise: smooth velocities -> physical qdd.
        raw = rng.normal(0.0, cfg.amp, size=(task.n_steps + cfg.smooth_window, n))
        w = cfg.smooth_window
        kernel = np.ones(w) / w
        smooth = np.vstack([np.convolve(raw[:, j], kernel, mode="valid") for j in range(n)]).T
        # Renormalize so the smoothed amplitude is comparable to cfg.amp.
        smooth *= cfg.amp / (np.std(smooth) + 1e-12)
        return lambda k: smooth[min(k, smooth.shape[0] - 1)]
    if policy == "fourier_nullspace":
        a = rng.normal(0.0, cfg.amp, size=(cfg.fourier_modes, n))
        b = rng.normal(0.0, cfg.amp, size=(cfg.fourier_modes, n))
        w = 2.0 * np.pi * cfg.base_freq

        def fn(k):
            t = k * task.dt
            out = np.zeros(n)
            for mdx in range(cfg.fourier_modes):
                wk = (mdx + 1) * w
                out += a[mdx] * np.sin(wk * t) + b[mdx] * np.cos(wk * t)
            return out

        return fn
    raise ValueError(f"unknown policy: {policy!r}")


def rollout(
    model: PlanarNRModel,
    task: Task,
    policy: str,
    cfg: NullspaceConfig | None = None,
    seed: int = 0,
) -> dict:
    """Run a task-compatible rollout. Returns states (n_steps x 3n) and diagnostics."""
    cfg = cfg or NullspaceConfig()
    n = model.n_links
    xi_fn = _excitation(policy, n, task, cfg, seed)

    q = np.asarray(task.q_init, float).copy()
    qs = np.zeros((task.n_steps, n))
    qds = np.zeros((task.n_steps, n))
    ee_err = np.zeros(task.n_steps)
    null_eff = np.zeros(task.n_steps)

    for k in range(task.n_steps):
        p_des, pdot_des = task.p_des(k, model)
        J = jacobian(q, model)
        Jp = np.linalg.pinv(J, rcond=cfg.rcond)
        N = np.eye(n) - Jp @ J
        p_ee = forward_kinematics(q, model)
        xdot_cmd = pdot_des + task.kp * (p_des - p_ee)
        ns_vel = N @ xi_fn(k)
        qd = Jp @ xdot_cmd + ns_vel

        qs[k] = q
        qds[k] = qd
        ee_err[k] = float(np.linalg.norm(p_ee - p_des))
        null_eff[k] = float(np.linalg.norm(ns_vel))
        q = q + qd * task.dt

    qdds = np.gradient(qds, task.dt, axis=0)
    states = np.hstack([qs, qds, qdds])
    return {
        "states": states,  # (n_steps x 3n): [q | qd | qdd]
        "ee_error": ee_err,
        "ee_error_mean": float(np.mean(ee_err)),
        "ee_error_max": float(np.max(ee_err)),
        "nullspace_effort": null_eff,
        "nullspace_effort_mean": float(np.mean(null_eff)),
        "policy": policy,
        "task_kind": task.kind,
    }
