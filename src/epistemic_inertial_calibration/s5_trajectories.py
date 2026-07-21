"""S5-1: executable excitation trajectory design (2R).

Every S5 Layer-1 policy emits ONE continuous, executable reference trajectory
(q_des, qd_des, qdd_des sampled at the control period dt), built by chaining
quintic segments that are C2-continuous at the junctions (junction
accelerations are zero by construction). Candidate segments are checked at
design time against the admissible envelope

    |q| <= q_range,  |qd| <= vel_limit,  |qdd| <= acc_limit,

and against a nominal-model torque budget |Y_ext beta_nominal| <=
torque_margin * tau_limit. The torque check uses the NOMINAL (controller)
parameters, never the true plant parameters: on hardware the true parameters
are unknown at design time.

Policies:
  hold_sequence     : point-to-point moves to random configurations with
                      dwell holds (the executable analogue of static data).
  smooth_random     : chain of random quintic segments (random target
                      position and velocity per segment).
  fourier_envelope  : one box-fitted Fourier trajectory (amplitude saturates
                      the position range, time dilation saturates the tighter
                      of the velocity/acceleration bounds; slowed further if
                      the nominal torque budget requires), entered via a
                      quintic transition from the start state.
  segment_fim_greedy: greedy selection among candidate segments by design-
                      time Fisher information (no prior, no feedback).
  segment_active_ig : greedy selection by information gain against the
                      CURRENT posterior precision, supplied by the caller;
                      in the closed-loop experiment the precision comes from
                      measured data, making this a genuine active loop.

The designers are deliberately reference-level: execution (tracking on the
friction plant, measurement, identification) lives in s5_execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .s5_model import FrictionModel2R, stacked_regressor_ext
from .trajectories import WindowConfig, fourier_window

SEGMENT_POLICIES = (
    "hold_sequence",
    "smooth_random",
    "fourier_envelope",
    "segment_fim_greedy",
    "segment_active_ig",
)


@dataclass(frozen=True)
class S5TrajConfig:
    """Design-time settings shared by all policies."""

    dt: float = 2e-3
    seg_duration: float = 0.6      # s per quintic segment
    n_segments: int = 20           # total reference duration = n_segments * seg_duration
    dwell_fraction: float = 0.5    # hold_sequence: fraction of each segment spent dwelling
    q_range: float = np.pi
    vel_limit: float = 2.0
    acc_limit: float = 4.0
    tau_limit: float = 40.0        # matches the Layer-3 default torque limit
    torque_margin: float = 0.8     # design-time nominal-torque budget fraction
    target_vel_frac: float = 0.7   # sample segment end velocities within this fraction
    n_candidates: int = 8          # candidates per greedy round
    max_resample: int = 20         # envelope-rejection retries per segment
    ridge: float = 1e-9            # fim_greedy score regularizer
    q_start: tuple[float, float] = (0.2, 0.3)
    torque_check_stride: int = 3   # design-time torque check subsampling
    design_stride: int = 5         # design-time Fisher-score subsampling

    @property
    def seg_steps(self) -> int:
        return max(2, int(round(self.seg_duration / self.dt)))

    @property
    def total_steps(self) -> int:
        return self.seg_steps * self.n_segments


# ---------------------------------------------------------------------------
# Quintic segments (boundary accelerations zero -> C2 chain)
# ---------------------------------------------------------------------------


def quintic_segment(
    q0: np.ndarray, qd0: np.ndarray, q1: np.ndarray, qd1: np.ndarray,
    n_steps: int, dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-joint quintic from (q0, qd0, qdd=0) to (q1, qd1, qdd=0) over n_steps*dt.

    Returns (q, qd, qdd), each (n_steps x 2), sampled at t = dt, 2dt, ..., T.
    The segment START state (t=0) is the previous segment's end and is not
    re-emitted, so chained segments concatenate without duplicate samples.
    """
    T = n_steps * dt
    q0 = np.asarray(q0, float)
    qd0 = np.asarray(qd0, float)
    dq = np.asarray(q1, float) - q0
    v0 = qd0
    v1 = np.asarray(qd1, float)
    # Quintic coefficients with qdd0 = qdd1 = 0.
    a0, a1, a2 = q0, v0, np.zeros_like(q0)
    a3 = (20.0 * dq - (8.0 * v1 + 12.0 * v0) * T) / (2.0 * T**3)
    a4 = (-30.0 * dq + (14.0 * v1 + 16.0 * v0) * T) / (2.0 * T**4)
    a5 = (12.0 * dq - 6.0 * (v1 + v0) * T) / (2.0 * T**5)
    t = (np.arange(1, n_steps + 1) * dt)[:, None]  # (n,1)
    q = a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5
    qd = a1 + 2 * a2 * t + 3 * a3 * t**2 + 4 * a4 * t**3 + 5 * a5 * t**4
    qdd = 2 * a2 + 6 * a3 * t + 12 * a4 * t**2 + 20 * a5 * t**3
    return q, qd, qdd


def _within_envelope(
    q: np.ndarray, qd: np.ndarray, qdd: np.ndarray,
    cfg: S5TrajConfig, fmodel: FrictionModel2R, beta_nominal: np.ndarray,
) -> bool:
    """Design-time check: state box + nominal-model torque budget."""
    if np.max(np.abs(q)) > cfg.q_range:
        return False
    if np.max(np.abs(qd)) > cfg.vel_limit:
        return False
    if np.max(np.abs(qdd)) > cfg.acc_limit:
        return False
    states = np.hstack([q, qd, qdd])[:: cfg.torque_check_stride]
    tau = (stacked_regressor_ext(states, fmodel) @ beta_nominal).reshape(-1, 2)
    return bool(np.max(np.abs(tau)) <= cfg.torque_margin * cfg.tau_limit)


def _sample_segment(
    q0: np.ndarray, qd0: np.ndarray, rng: np.random.Generator,
    cfg: S5TrajConfig, fmodel: FrictionModel2R, beta_nominal: np.ndarray,
    end_at_rest: bool = False,
    n_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rejection-sample one admissible segment from (q0, qd0).

    Retries with progressively smaller displacements; the final fallback (a
    slow-down-in-place segment to rest) is admissible for any start state
    inside the envelope.
    """
    n = cfg.seg_steps if n_steps is None else n_steps
    for trial in range(cfg.max_resample):
        shrink = 0.8**trial
        q1 = rng.uniform(-cfg.q_range, cfg.q_range, 2) * shrink + (1 - shrink) * q0
        qd1 = (
            np.zeros(2)
            if end_at_rest
            else rng.uniform(-1.0, 1.0, 2) * cfg.target_vel_frac * cfg.vel_limit * shrink
        )
        seg = quintic_segment(q0, qd0, q1, qd1, n, cfg.dt)
        if _within_envelope(*seg, cfg, fmodel, beta_nominal):
            return seg
    return quintic_segment(q0, qd0, q0, np.zeros(2), n, cfg.dt)  # stop in place


# ---------------------------------------------------------------------------
# Fourier envelope trajectory (continuous, box- and torque-fitted)
# ---------------------------------------------------------------------------


_S_DENSE = np.linspace(0.0, 2.0, 400)


def _fit_fourier(
    rng: np.random.Generator, cfg: S5TrajConfig,
    fmodel: FrictionModel2R, beta_nominal: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Draw a unit Fourier shape and fit (amp, lam) to the envelope + torque budget.

    The unit shape (fundamental 1 Hz, all harmonics integer multiples) is
    periodic with period 1 s, so any phase is a valid entry point. Amplitude
    saturates the position range; the time dilation lam saturates the tighter
    of the velocity/acceleration bounds, then is reduced until the nominal
    torque budget holds (dense probe over one full 2 s cycle).
    """
    unit_cfg = WindowConfig(fourier_amp=1.0, fourier_base_freq=1.0, fourier_horizon=2.0)
    dense = fourier_window(rng, _S_DENSE.size, unit_cfg)
    amp = cfg.q_range / max(np.max(np.abs(dense[:, 0:2])), 1e-12)
    lam_v = cfg.vel_limit / max(amp * np.max(np.abs(dense[:, 2:4])), 1e-12)
    lam_a = float(np.sqrt(cfg.acc_limit / max(amp * np.max(np.abs(dense[:, 4:6])), 1e-12)))
    lam = min(lam_v, lam_a)
    for _ in range(30):
        probe = np.hstack([
            amp * dense[:, 0:2],
            amp * lam * dense[:, 2:4],
            amp * lam**2 * dense[:, 4:6],
        ])[:: cfg.torque_check_stride]
        tau = (stacked_regressor_ext(probe, fmodel) @ beta_nominal).reshape(-1, 2)
        if np.max(np.abs(tau)) <= cfg.torque_margin * cfg.tau_limit:
            break
        lam *= 0.85  # slow down until the nominal torque budget holds
    return dense, amp, lam


def _fourier_profile(
    dense: np.ndarray, amp: float, lam: float, s0: float,
    n_steps: int, dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the fitted periodic Fourier trajectory from phase s0 at period dt."""
    t = np.arange(1, n_steps + 1) * dt
    s = (s0 + lam * t) % 2.0
    q = amp * np.vstack([np.interp(s, _S_DENSE, dense[:, j]) for j in (0, 1)]).T
    qd = amp * lam * np.vstack([np.interp(s, _S_DENSE, dense[:, j]) for j in (2, 3)]).T
    qdd = amp * lam**2 * np.vstack([np.interp(s, _S_DENSE, dense[:, j]) for j in (4, 5)]).T
    return q, qd, qdd


def _fourier_entry(dense: np.ndarray, amp: float, lam: float, s0: float):
    """Entry state (q, qd) of the fitted trajectory at phase s0."""
    q = amp * np.array([np.interp(s0, _S_DENSE, dense[:, j]) for j in (0, 1)])
    qd = amp * lam * np.array([np.interp(s0, _S_DENSE, dense[:, j]) for j in (2, 3)])
    return q, qd


# ---------------------------------------------------------------------------
# Reference designers
# ---------------------------------------------------------------------------


@dataclass
class DesignContext:
    """Scoring context for the greedy designers.

    precision: callable returning the CURRENT base-coordinate precision matrix
        (posterior precision for segment_active_ig; the closed-loop experiment
        updates it from measured data between segments).
    V_base: (rank x 10) base projection used for design-time scoring.
    sigma_tau: assumed torque noise for the Fisher information scale.
    """

    V_base: np.ndarray
    sigma_tau: float
    precision: object = None  # () -> (rank x rank); None for fim_greedy
    F_acc: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.F_acc is None:
            self.F_acc = np.zeros((self.V_base.shape[0], self.V_base.shape[0]))


def _segment_fisher(
    seg, ctx: DesignContext, fmodel: FrictionModel2R, stride: int
) -> np.ndarray:
    states = np.hstack(seg)[::stride]
    Wb = stacked_regressor_ext(states, fmodel) @ ctx.V_base.T
    return (1.0 / ctx.sigma_tau**2) * (Wb.T @ Wb)


def design_next_segment(
    policy: str,
    q0: np.ndarray,
    qd0: np.ndarray,
    seg_idx: int,
    rng: np.random.Generator,
    cfg: S5TrajConfig,
    fmodel: FrictionModel2R,
    beta_nominal: np.ndarray,
    ctx: DesignContext | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Design the next reference segment from the (reference) state (q0, qd0)."""
    if policy == "hold_sequence":
        move_steps = max(2, int(cfg.seg_steps * (1.0 - cfg.dwell_fraction)))
        dwell_steps = cfg.seg_steps - move_steps
        qm, qdm, qddm = _sample_segment(
            q0, qd0, rng, cfg, fmodel, beta_nominal,
            end_at_rest=True, n_steps=move_steps,
        )
        if dwell_steps > 0:
            q_hold = np.tile(qm[-1], (dwell_steps, 1))
            zeros = np.zeros((dwell_steps, 2))
            return (
                np.vstack([qm, q_hold]),
                np.vstack([qdm, zeros]),
                np.vstack([qddm, zeros]),
            )
        return qm, qdm, qddm
    if policy == "smooth_random":
        return _sample_segment(q0, qd0, rng, cfg, fmodel, beta_nominal)
    if policy in ("segment_fim_greedy", "segment_active_ig"):
        if ctx is None:
            raise ValueError(f"{policy} requires a DesignContext")
        cands = [
            _sample_segment(q0, qd0, rng, cfg, fmodel, beta_nominal)
            for _ in range(cfg.n_candidates)
        ]
        F_list = [_segment_fisher(c, ctx, fmodel, cfg.design_stride) for c in cands]
        dim = ctx.V_base.shape[0]
        if policy == "segment_fim_greedy":
            base = ctx.F_acc + cfg.ridge * np.eye(dim)
        else:
            base = ctx.precision()
        scores = [np.linalg.slogdet(base + F)[1] for F in F_list]
        j = int(np.argmax(scores))
        ctx.F_acc = ctx.F_acc + F_list[j]
        return cands[j]
    raise ValueError(f"unknown policy: {policy!r}")


def design_reference(
    policy: str,
    rng: np.random.Generator,
    cfg: S5TrajConfig,
    fmodel: FrictionModel2R,
    beta_nominal: np.ndarray,
    ctx: DesignContext | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Design the full open-loop reference for a non-adaptive policy.

    fourier_envelope designs a transition segment plus one continuous Fourier
    trajectory; the segment policies chain segments from the reference
    endpoint. segment_active_ig should instead be driven segment-by-segment by
    the closed-loop experiment (its scoring depends on measured data).
    """
    q0 = np.asarray(cfg.q_start, float)
    qd0 = np.zeros(2)
    if policy == "fourier_envelope":
        # Fit the periodic Fourier trajectory once, then enter it at the phase
        # whose position is closest to the start state (the trajectory is
        # periodic, so any entry phase is valid). Grow the transition quintic
        # until it fits the envelope; if even the longest transition fails,
        # slow the Fourier down and retry.
        dense, amp, lam = _fit_fourier(rng, cfg, fmodel, beta_nominal)
        n_max_trans = cfg.total_steps - cfg.seg_steps  # keep >= 1 segment of Fourier
        for _ in range(10):
            s_grid = np.linspace(0.0, 2.0, 64, endpoint=False)
            dists = [
                float(np.max(np.abs(_fourier_entry(dense, amp, lam, s)[0] - q0)))
                for s in s_grid
            ]
            s0 = float(s_grid[int(np.argmin(dists))])
            q_e, qd_e = _fourier_entry(dense, amp, lam, s0)
            n_trans = cfg.seg_steps
            fitted = None
            while n_trans <= n_max_trans:
                qt, qdt, qddt = quintic_segment(q0, qd0, q_e, qd_e, n_trans, cfg.dt)
                if _within_envelope(qt, qdt, qddt, cfg, fmodel, beta_nominal):
                    fitted = (qt, qdt, qddt)
                    break
                n_trans = max(n_trans + 1, int(n_trans * 1.5))
            if fitted is not None:
                break
            lam *= 0.8  # entry velocity too demanding: slow the Fourier down
        if fitted is None:  # pragma: no cover - safeguard
            fitted = quintic_segment(q0, qd0, q_e, qd_e, n_max_trans, cfg.dt)
            n_trans = n_max_trans
        qt, qdt, qddt = fitted
        n_rest = cfg.total_steps - qt.shape[0]
        qf, qdf, qddf = _fourier_profile(dense, amp, lam, s0, n_rest, cfg.dt)
        return (
            np.vstack([qt, qf]),
            np.vstack([qdt, qdf]),
            np.vstack([qddt, qddf]),
        )
    qs, qds, qdds = [], [], []
    for k in range(cfg.n_segments):
        seg = design_next_segment(
            policy, q0, qd0, k, rng, cfg, fmodel, beta_nominal, ctx=ctx
        )
        qs.append(seg[0])
        qds.append(seg[1])
        qdds.append(seg[2])
        q0, qd0 = seg[0][-1], seg[1][-1]
    return np.vstack(qs), np.vstack(qds), np.vstack(qdds)
