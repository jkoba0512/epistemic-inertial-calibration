"""S2-A: end-effector tasks (hold / reach) for the 4R Layer-2 experiments.

A task defines the desired end-effector trajectory p_des(t) (and its velocity) plus
a fixed, singularity-avoiding initial joint configuration. The null-space excitation
(task_policies) rides on top without moving the end-effector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .planar_nr import PlanarNRModel, forward_kinematics

# Fixed initial configuration away from singularities (links not collinear/stretched).
DEFAULT_Q_INIT: tuple[float, ...] = (0.3, 0.6, -0.5, 0.7)


@dataclass(frozen=True)
class Task:
    """End-effector task.

    kind: "hold" (stay at FK(q_init)) or "reach" (move to p_goal over the rollout).
    q_init: fixed initial joint configuration.
    p_goal: target EE position for "reach" (ignored for "hold").
    n_steps, dt: rollout length and timestep.
    tolerance: acceptable EE position error.
    kp: task-space position-feedback gain used to cancel discrete-integration drift
        (null-space strictness holds at the velocity level; kp keeps EE on target).
    """

    kind: str = "hold"
    q_init: tuple[float, ...] = DEFAULT_Q_INIT
    p_goal: tuple[float, ...] | None = None
    n_steps: int = 200
    dt: float = 0.01
    tolerance: float = 1e-2
    kp: float = 20.0

    def p_start(self, model: PlanarNRModel) -> np.ndarray:
        return forward_kinematics(self.q_init, model)

    def _s(self, k: int) -> tuple[float, float]:
        """Minimum-jerk time scaling s(t) in [0,1] and its derivative (per step k)."""
        T = (self.n_steps - 1) * self.dt
        t = k * self.dt
        if T <= 0:
            return 1.0, 0.0
        u = min(max(t / T, 0.0), 1.0)
        s = 10 * u**3 - 15 * u**4 + 6 * u**5
        sdot = (30 * u**2 - 60 * u**3 + 30 * u**4) / T
        return s, sdot

    def p_des(self, k: int, model: PlanarNRModel) -> tuple[np.ndarray, np.ndarray]:
        """Desired EE position and velocity at step k."""
        p0 = self.p_start(model)
        if self.kind == "hold":
            return p0, np.zeros(2)
        if self.kind == "reach":
            if self.p_goal is None:
                raise ValueError("reach task requires p_goal")
            pg = np.asarray(self.p_goal, float)
            s, sdot = self._s(k)
            return p0 + s * (pg - p0), sdot * (pg - p0)
        raise ValueError(f"unknown task kind: {self.kind!r}")


def default_reach_goal(model: PlanarNRModel, offset=(0.15, 0.15)) -> tuple[float, ...]:
    """A reachable goal near the start pose (kept easy for S2-A)."""
    return tuple(forward_kinematics(DEFAULT_Q_INIT, model) + np.asarray(offset, float))
