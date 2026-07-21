"""S5-5: iiwa 7-DoF generalization of the hardware-executable pipeline.

Friction-extended iiwa model (Pinocchio rigid regressor + per-joint viscous
and smoothed-Coulomb friction, 70 + 14 = 84 parameters), closed-loop
execution with an inertia-weighted bounded-authority FF+PD controller inside
the URDF limits, measured-data identification with velocity row filtering,
and the velocity-aware terminal-feasibility grid with fault-consistent
labels. Mirrors the 2R (s5_model/s5_execution) and 4R (s5_layer2) modules;
requires the `iiwa` extra (Pinocchio).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .iiwa_model import IiwaModel, load_iiwa
from .posterior import GaussianPosterior
from .evaluation import minimum_jerk_trajectory, stable_seed
from .s5_execution import estimate_qdd

# Drake per-joint acceleration limits recorded in the vendored URDF.
IIWA_ACC_LIMITS = np.array([8.57, 8.57, 8.74, 11.36, 12.23, 15.72, 15.72])

DEFAULT_FV = (0.50, 0.50, 0.30, 0.30, 0.20, 0.15, 0.10)
DEFAULT_FC = (1.00, 1.00, 0.70, 0.70, 0.40, 0.30, 0.20)
# Reflected rotor inertias (kg m^2). On highly geared arms these dominate the
# distal LINK inertias; a plant without them has a nearly massless last joint,
# which no realistic torque controller could regulate.
DEFAULT_ARMATURE = (0.15, 0.15, 0.12, 0.12, 0.10, 0.10, 0.08)


@dataclass
class FrictionIiwa:
    """iiwa with rotor (armature) inertia and viscous + smoothed-Coulomb friction."""

    base: IiwaModel
    fv: tuple[float, ...] = DEFAULT_FV
    fc: tuple[float, ...] = DEFAULT_FC
    armature: tuple[float, ...] = DEFAULT_ARMATURE
    eps: float = 0.05

    @property
    def nv(self) -> int:
        return self.base.nv

    @property
    def n_params(self) -> int:
        return self.base.n_params + 3 * self.nv  # 70 + 7 armature + 14 friction = 91

    def beta_ext_true(self) -> np.ndarray:
        fr = np.empty(2 * self.nv)
        fr[0::2] = self.fv
        fr[1::2] = self.fc
        return np.concatenate([
            self.base.beta_true(), np.asarray(self.armature, float), fr
        ])

    def friction_torque(self, qd: np.ndarray) -> np.ndarray:
        qd = np.asarray(qd, float)
        return np.asarray(self.fv) * qd + np.asarray(self.fc) * np.tanh(qd / self.eps)

    def forward_dynamics(self, q, qd, tau) -> np.ndarray:
        """Plant: (M(q) + diag(armature)) qdd = tau - h(q, qd) - tau_fric(qd)."""
        q = np.asarray(q, float)
        qd = np.asarray(qd, float)
        M = self.base.mass_matrix(q) + np.diag(self.armature)
        h = self.base.inverse_dynamics(q, qd, np.zeros(self.nv))  # Coriolis + gravity
        rhs = np.asarray(tau, float) - h - self.friction_torque(qd)
        return np.linalg.solve(M, rhs)


def load_friction_iiwa() -> FrictionIiwa:
    return FrictionIiwa(base=load_iiwa())


def regressor_ext_iiwa(q, qd, qdd, fm: FrictionIiwa) -> np.ndarray:
    """(7 x 91) extended regressor: [rigid Pinocchio | armature | friction]."""
    Y = fm.base.regressor(q, qd, qdd)
    qd = np.asarray(qd, float)
    qdd = np.asarray(qdd, float)
    n = fm.nv
    A = np.diag(qdd)  # armature block: tau_i += Ja_i * qdd_i
    F = np.zeros((n, 2 * n))
    for i in range(n):
        F[i, 2 * i] = qd[i]
        F[i, 2 * i + 1] = np.tanh(qd[i] / fm.eps)
    return np.hstack([Y, A, F])


def stacked_regressor_ext_iiwa(states: np.ndarray, fm: FrictionIiwa) -> np.ndarray:
    n = fm.nv
    return np.vstack([
        regressor_ext_iiwa(s[:n], s[n:2 * n], s[2 * n:3 * n], fm) for s in states
    ])


def sample_states_hw(fm: FrictionIiwa, n: int, seed: int) -> np.ndarray:
    """Random states INSIDE the hardware limits (position, velocity, drake acc)."""
    rng = np.random.default_rng(seed)
    nv = fm.nv
    vlim = np.array(fm.base.model.velocityLimit, float)
    q = np.vstack([fm.base.random_configuration(rng) for _ in range(n)])
    qd = rng.uniform(-vlim, vlim, size=(n, nv))
    qdd = rng.uniform(-IIWA_ACC_LIMITS, IIWA_ACC_LIMITS, size=(n, nv))
    return np.hstack([q, qd, qdd])


def nominal_beta_ext_iiwa(fm: FrictionIiwa, rng: np.random.Generator,
                          mass_error: float = 0.2, friction_error: float = 0.5) -> np.ndarray:
    """CAD-like nominal parameters: per-link scale error + friction errors.

    Each link's 10-parameter block is scaled by one factor (same geometry,
    wrong mass), which keeps the perturbed block physically consistent.
    """
    phi = fm.base.beta_true().copy()
    n_links = phi.size // 10
    for i in range(n_links):
        phi[10 * i:10 * (i + 1)] *= 1 + rng.uniform(-mass_error, mass_error)
    arm = np.asarray(fm.armature, float) * (1 + rng.uniform(-mass_error, mass_error, fm.nv))
    fr = np.empty(2 * fm.nv)
    fr[0::2] = np.asarray(fm.fv) * (1 + rng.uniform(-friction_error, friction_error, fm.nv))
    fr[1::2] = np.asarray(fm.fc) * (1 + rng.uniform(-friction_error, friction_error, fm.nv))
    return np.concatenate([phi, arm, fr])


# ---------------------------------------------------------------------------
# Closed-loop execution (inertia-weighted bounded-authority FF+PD)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionConfigIiwa:
    kp: float = 400.0   # omega = 20 rad/s, critically damped (iiwa-like stiffness);
    kd: float = 40.0    # weaker gains leave distal joints friction-dominated
    pd_authority: float = 0.3
    torque_scale: float = 1.0     # multiplies the URDF effort limits
    dt: float = 2e-3
    sigma_q: float = 1e-4
    sigma_qd: float = 1e-3
    sigma_tau: float = 1e-2
    vel_fault_margin: float = 1.0  # fault when |qd| exceeds margin * URDF limit
    # Wider acceleration-estimation window than the planar systems: the seven
    # coupled, heavy joints move at lower fundamental frequency, so a 21-sample
    # (42 ms) window leaves enough differentiated-noise in the estimated qdd to
    # bias the inertial columns (errors-in-variables). 51 samples (102 ms)
    # restores the physically expected ordering (free > null-space > hold).
    smooth_window: int = 51
    obs_decimate: int = 5
    edge_trim: int = 51
    qd_min: float = 0.05


def execute_reference_iiwa(
    fm: FrictionIiwa,
    q_des: np.ndarray, qd_des: np.ndarray, qdd_des: np.ndarray,
    beta_ctrl: np.ndarray,
    cfg: ExecutionConfigIiwa,
    rng: np.random.Generator,
) -> dict:
    nv = fm.nv
    limits = fm.base.effort_limit() * cfg.torque_scale
    vlim = np.array(fm.base.model.velocityLimit, float) * cfg.vel_fault_margin
    qlo = np.array(fm.base.model.lowerPositionLimit, float)
    qhi = np.array(fm.base.model.upperPositionLimit, float)
    n_steps = q_des.shape[0]
    q = q_des[0].copy()
    qd = qd_des[0].copy()

    # Full nominal mass matrix at the start pose (regressor column trick on
    # the nominal rigid parameters); frozen inertia weighting for the PD.
    # Extended-regressor column trick with the NOMINAL parameters: includes
    # the nominal armature automatically (friction columns vanish at qd = 0).
    zero = np.zeros(nv)
    g0 = regressor_ext_iiwa(q_des[0], zero, zero, fm) @ beta_ctrl
    M0 = np.empty((nv, nv))
    for j in range(nv):
        e = np.zeros(nv)
        e[j] = 1.0
        M0[:, j] = regressor_ext_iiwa(q_des[0], zero, e, fm) @ beta_ctrl - g0
    M0 = 0.5 * (M0 + M0.T)

    q_true = np.zeros((n_steps, nv))
    qd_true = np.zeros((n_steps, nv))
    q_meas = np.zeros((n_steps, nv))
    qd_meas = np.zeros((n_steps, nv))
    tau_meas = np.zeros((n_steps, nv))
    sat_hits = 0
    fault = False
    for k in range(n_steps):
        qm = q + rng.normal(0.0, cfg.sigma_q, nv)
        qdm = qd + rng.normal(0.0, cfg.sigma_qd, nv)
        tau_ff = regressor_ext_iiwa(qm, qdm, qdd_des[k], fm) @ beta_ctrl
        tau_pd = np.clip(
            M0 @ (cfg.kp * (q_des[k] - qm) + cfg.kd * (qd_des[k] - qdm)),
            -cfg.pd_authority * limits, cfg.pd_authority * limits,
        )
        tau_sat = np.clip(tau_ff + tau_pd, -limits, limits)
        sat_hits += int(np.sum(np.abs(tau_ff + tau_pd - tau_sat) > 1e-9))
        qdd = fm.forward_dynamics(q, qd, tau_sat)
        qd = qd + qdd * cfg.dt
        q = q + qd * cfg.dt
        q_true[k] = q
        qd_true[k] = qd
        q_meas[k] = qm
        qd_meas[k] = qdm
        tau_meas[k] = tau_sat + rng.normal(0.0, cfg.sigma_tau, nv)
        if (np.any(np.abs(qd) > vlim) or np.any(q < qlo) or np.any(q > qhi)
                or not np.all(np.isfinite(q))):
            fault = True
            m = k + 1
            q_true, qd_true = q_true[:m], qd_true[:m]
            q_meas, qd_meas, tau_meas = q_meas[:m], qd_meas[:m], tau_meas[:m]
            q_des = q_des[:m]
            break
    track = q_true - q_des[: q_true.shape[0]]
    return {
        "q_true": q_true, "qd_true": qd_true,
        "q_meas": q_meas, "qd_meas": qd_meas, "tau_meas": tau_meas,
        "fault": fault,
        "saturation_fraction": sat_hits / max(1, q_true.shape[0] * nv),
        "tracking_rmse": float(np.sqrt(np.mean(track**2))),
        "n_steps": q_true.shape[0],
    }


def build_observations_iiwa(
    execu: dict, V_base: np.ndarray, fm: FrictionIiwa, cfg: ExecutionConfigIiwa
) -> tuple[np.ndarray, np.ndarray]:
    qdd_est = estimate_qdd(execu["qd_meas"], cfg.dt, cfg.smooth_window)
    states = np.hstack([execu["q_meas"], execu["qd_meas"], qdd_est])
    t0, t1 = cfg.edge_trim, states.shape[0] - cfg.edge_trim
    keep = np.arange(t0, t1, cfg.obs_decimate)
    states = states[keep]
    W_base = stacked_regressor_ext_iiwa(states, fm) @ V_base.T
    y = execu["tau_meas"][keep].reshape(-1)
    nv = fm.nv
    row_qd = np.abs(states[:, nv:2 * nv]).reshape(-1)
    mask = row_qd >= cfg.qd_min
    return W_base[mask], y[mask]


def posterior_from_observations(
    W_base: np.ndarray, y: np.ndarray, a_prior: np.ndarray,
    prior_std: float, sigma_tau: float,
) -> tuple[GaussianPosterior, float]:
    """Two-pass residual-noise posterior (see s5_layer1)."""
    post = GaussianPosterior.isotropic_prior(a_prior, prior_std=prior_std)
    if y.size == 0:
        return post, sigma_tau
    pass1 = GaussianPosterior.isotropic_prior(a_prior, prior_std=prior_std)
    pass1.update(W_base, y, sigma_tau)
    resid = y - W_base @ pass1.mean()
    sigma_eff = float(max(sigma_tau, np.sqrt(np.mean(resid**2))))
    post.update(W_base, y, sigma_eff)
    return post, sigma_eff


# ---------------------------------------------------------------------------
# Layer-2: task-compatible null-space excitation, executed
# ---------------------------------------------------------------------------

Q_START = (0.0, 0.3, 0.0, -0.8, 0.0, 0.6, 0.0)
IIWA_LAYER2_POLICIES = ("free_reference", "no_exploration", "nullspace_fourier")


def _nullspace_reference(fm: FrictionIiwa, q_start, n_steps: int, dt: float,
                         amp: float, seed: int, kp_task: float = 30.0,
                         modes: int = 5, base_freq: float = 0.5):
    """Kinematic resolved-rate hold reference with Fourier null-space excitation."""
    nv = fm.nv
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, amp, size=(modes, nv))
    b = rng.normal(0.0, amp, size=(modes, nv))
    w = 2.0 * np.pi * base_freq
    t = np.arange(n_steps) * dt
    prof = np.zeros((n_steps, nv))
    for m in range(modes):
        wk = (m + 1) * w
        prof += np.outer(np.sin(wk * t), a[m]) + np.outer(np.cos(wk * t), b[m])
    vlim = np.array(fm.base.model.velocityLimit, float)
    for _ in range(10):
        q = np.asarray(q_start, float).copy()
        p0 = fm.base.forward_kinematics(q_start)
        qs = np.zeros((n_steps, nv))
        qds = np.zeros((n_steps, nv))
        for k in range(n_steps):
            J = fm.base.position_jacobian(q)
            Jp = np.linalg.pinv(J, rcond=1e-10)
            N = np.eye(nv) - Jp @ J
            qd = Jp @ (kp_task * (p0 - fm.base.forward_kinematics(q))) + N @ prof[k]
            qs[k] = q
            qds[k] = qd
            q = q + qd * dt
        # Velocity-limit check on the designed reference. The 60% margin
        # leaves headroom for tracking overshoot on the light distal joints,
        # where the friction-compensation error alone can transiently double
        # the realized velocity.
        ratio = float(np.max(np.abs(qds) / vlim[None, :]))
        if ratio <= 0.6 or amp == 0.0:
            break
        prof = prof * (0.55 / ratio)
    qdd = np.gradient(qds, dt, axis=0)
    return qs, qds, qdd


def _free_reference_iiwa(fm: FrictionIiwa, n_steps: int, dt: float, q_start,
                         beta_nominal: np.ndarray, cfg: ExecutionConfigIiwa,
                         rng: np.random.Generator, torque_margin: float = 0.6,
                         lam_cap: float = 0.8):
    """Box/torque-fitted periodic Fourier free excitation centered at q_start."""
    nv = fm.nv
    vlim = np.array(fm.base.model.velocityLimit, float)
    qlo = np.array(fm.base.model.lowerPositionLimit, float)
    qhi = np.array(fm.base.model.upperPositionLimit, float)
    q_c = np.asarray(q_start, float)
    head = np.minimum(qhi - q_c, q_c - qlo)  # per-joint position headroom

    K = 5
    s_dense = np.linspace(0.0, 2.0, 400)
    a = rng.normal(0.0, 1.0, size=(K, nv))
    b = rng.normal(0.0, 1.0, size=(K, nv))
    q_u = np.zeros((s_dense.size, nv))
    qd_u = np.zeros_like(q_u)
    qdd_u = np.zeros_like(q_u)
    for k in range(1, K + 1):
        w = 2.0 * np.pi * k
        sk = np.sin(w * s_dense)[:, None]
        ck = np.cos(w * s_dense)[:, None]
        q_u += (a[k - 1] * sk + b[k - 1] * ck) / k
        qd_u += (a[k - 1] * w * ck - b[k - 1] * w * sk) / k
        qdd_u += (-a[k - 1] * w**2 * sk - b[k - 1] * w**2 * ck) / k
    # Per-joint amplitude to the position headroom (90% of it).
    amp_j = 0.9 * head / np.clip(np.max(np.abs(q_u), axis=0), 1e-12, None)
    limits = fm.base.effort_limit() * cfg.torque_scale * torque_margin

    def profile(lam_, stride=4):
        return np.hstack([
            q_c + amp_j * q_u, (amp_j * qd_u) * lam_, (amp_j * qdd_u) * lam_**2
        ])[::stride]

    for _ in range(8):
        lam_v = float(np.min(vlim / np.clip(
            np.max(np.abs(amp_j * qd_u), axis=0), 1e-12, None)))
        lam_a = float(np.sqrt(np.min(IIWA_ACC_LIMITS / np.clip(
            np.max(np.abs(amp_j * qdd_u), axis=0), 1e-12, None))))
        lam = min(lam_v, lam_a, lam_cap)
        ok = False
        for _ in range(20):
            tau = (stacked_regressor_ext_iiwa(profile(lam), fm)
                   @ np.asarray(beta_nominal, float)).reshape(-1, nv)
            if np.all(np.max(np.abs(tau), axis=0) <= limits):
                ok = True
                break
            lam *= 0.9
        if ok:
            break
        amp_j = amp_j * 0.8

    # Enter the periodic trajectory at the phase closest to q_start, and grow
    # the transition quintic until it fits the per-joint velocity/acceleration
    # limits (the entry displacement and velocity set its internal peaks).
    from .s5_trajectories import quintic_segment

    def entry(s0):
        qe = q_c + amp_j * np.array([np.interp(s0, s_dense, q_u[:, j]) for j in range(nv)])
        qde = amp_j * lam * np.array([np.interp(s0, s_dense, qd_u[:, j]) for j in range(nv)])
        return qe, qde

    s_grid = np.linspace(0.0, 2.0, 64, endpoint=False)
    s0 = float(s_grid[int(np.argmin([
        np.max(np.abs(entry(s)[0] - q_c)) for s in s_grid
    ]))])
    n_max_trans = n_steps // 2
    for _ in range(8):
        q_e, qd_e = entry(s0)
        # Required transition duration from the min-jerk peak factors
        # (|qd|peak ~ 1.875 d/T, |qdd|peak ~ 5.77 d/T^2), with 25% margin.
        d = np.abs(q_e - q_c)
        T_req = 1.25 * float(np.max(np.maximum(
            1.875 * d / vlim, np.sqrt(5.77 * d / IIWA_ACC_LIMITS)
        )))
        n_trans = max(2, int(np.ceil(T_req / dt)))
        if n_trans <= n_max_trans:
            qt, qdt, qddt = quintic_segment(q_c, np.zeros(nv), q_e, qd_e, n_trans, dt)
            if (np.all(np.abs(qdt) <= vlim[None, :]) and
                    np.all(np.abs(qddt) <= IIWA_ACC_LIMITS[None, :])):
                break
        amp_j = amp_j * 0.8  # entry too far/fast for the budget: shrink
    else:  # pragma: no cover - safeguard
        qt, qdt, qddt = quintic_segment(q_c, np.zeros(nv), q_e, qd_e, n_max_trans, dt)
    n_rest = n_steps - qt.shape[0]
    t = np.arange(1, n_rest + 1) * dt
    s = (s0 + lam * t) % 2.0
    q = q_c + amp_j * np.vstack([np.interp(s, s_dense, q_u[:, j]) for j in range(nv)]).T
    qd = amp_j * lam * np.vstack([np.interp(s, s_dense, qd_u[:, j]) for j in range(nv)]).T
    qdd = amp_j * lam**2 * np.vstack([np.interp(s, s_dense, qdd_u[:, j]) for j in range(nv)]).T
    return np.vstack([qt, q]), np.vstack([qdt, qd]), np.vstack([qddt, qdd])


def run_iiwa_layer2(
    fm: FrictionIiwa, V_base: np.ndarray, n_seeds: int,
    cfg: ExecutionConfigIiwa | None = None,
    n_steps: int = 2000, prior_std: float = 0.15,
    nullspace_amp: float = 0.4,
) -> dict:
    cfg = cfg or ExecutionConfigIiwa()
    alpha_true = V_base @ fm.beta_ext_true()
    hold = sample_states_hw(fm, 600, seed=9991)
    W_holdout = stacked_regressor_ext_iiwa(hold, fm) @ V_base.T

    by_seed = []
    for seed in range(n_seeds):
        rng_nom = np.random.default_rng([seed, 5])
        beta_ctrl = nominal_beta_ext_iiwa(fm, rng_nom)
        for policy in IIWA_LAYER2_POLICIES:
            if policy == "free_reference":
                ref = _free_reference_iiwa(
                    fm, n_steps, cfg.dt, Q_START, beta_ctrl, cfg,
                    np.random.default_rng([seed, 21]),
                )
            elif policy == "no_exploration":
                ref = _nullspace_reference(fm, Q_START, n_steps, cfg.dt, 0.0, seed)
            else:
                ref = _nullspace_reference(fm, Q_START, n_steps, cfg.dt,
                                           nullspace_amp, seed)
            execu = execute_reference_iiwa(
                fm, *ref, beta_ctrl, cfg, np.random.default_rng([seed, 3])
            )
            W_base, y = build_observations_iiwa(execu, V_base, fm, cfg)
            a_prior = V_base @ beta_ctrl
            post, sigma_eff = posterior_from_observations(
                W_base, y, a_prior, prior_std, cfg.sigma_tau
            )
            mu = post.mean()
            p0 = fm.base.forward_kinematics(np.asarray(Q_START, float))
            ee = [
                float(np.linalg.norm(fm.base.forward_kinematics(qk) - p0))
                for qk in execu["q_true"][:: max(1, execu["q_true"].shape[0] // 100)]
            ]
            by_seed.append({
                "policy": policy, "seed": seed,
                "logdet_cov": post.logdet_cov(),
                "alpha_rmse": float(np.sqrt(np.mean((mu - alpha_true) ** 2))),
                "holdout_torque_rmse": float(np.sqrt(np.mean(
                    (W_holdout @ (mu - alpha_true)) ** 2))),
                "sigma_eff": sigma_eff,
                "ee_error_max": float(np.max(ee)) if policy != "free_reference" else float("nan"),
                "tracking_rmse": execu["tracking_rmse"],
                "fault": bool(execu["fault"]),
                "max_abs_qd": float(np.max(np.abs(execu["qd_true"]))),
                "saturation_fraction": execu["saturation_fraction"],
            })

    keys = ["logdet_cov", "alpha_rmse", "holdout_torque_rmse", "sigma_eff",
            "ee_error_max", "tracking_rmse", "max_abs_qd", "saturation_fraction"]

    def agg(v):
        arr = np.array([x for x in v if np.isfinite(x)], float)
        return ({"mean": float(arr.mean()), "std": float(arr.std())}
                if arr.size else {"mean": float("nan"), "std": float("nan")})

    summary = {}
    for p in IIWA_LAYER2_POLICIES:
        rows = [r for r in by_seed if r["policy"] == p]
        summary[p] = {
            "n_seeds": len(rows),
            "n_faults": int(sum(r["fault"] for r in rows)),
            **{k: agg([r[k] for r in rows]) for k in keys},
        }
    return {"by_seed": by_seed, "summary": summary}


# ---------------------------------------------------------------------------
# Layer-3: velocity-aware terminal feasibility with fault labels
# ---------------------------------------------------------------------------

CAL_SCALES = {"cal0": 0.0, "cal_small": 0.05, "cal_medium": 0.15, "cal_large": 0.3}
TORQUE_SCALES = {"loose": 1.0, "medium": 0.3, "tight": 0.12}
HORIZONS = {"normal": 1500, "short": 700}
TARGETS = {
    "near": (0.4, 0.5, 0.2, -1.0, 0.1, 0.8, 0.2),
    "far": (1.5, 1.0, 0.8, -1.8, 0.7, 1.2, 0.6),
}


@dataclass(frozen=True)
class IiwaL3Config:
    kp: float = 100.0
    kd: float = 20.0
    pd_authority: float = 0.3
    torque_scale: float = 1.0
    dt: float = 1e-3
    n_steps: int = 1500
    settle_frac: float = 0.2
    pos_threshold: float = 0.03   # m, EE hold error
    vel_threshold: float = 0.10   # rad/s


def feasibility_components_iiwa(
    fm: FrictionIiwa, beta_ext: np.ndarray, q_start, q_goal, cfg: IiwaL3Config,
    stride: int = 10,
) -> dict:
    q_des, qd_des, qdd_des = minimum_jerk_trajectory(
        q_start, q_goal, cfg.n_steps, cfg.dt
    )
    limits = fm.base.effort_limit() * cfg.torque_scale
    vlim = np.array(fm.base.model.velocityLimit, float)
    states = np.hstack([q_des, qd_des, qdd_des])[::stride]
    tau = (stacked_regressor_ext_iiwa(states, fm)
           @ np.asarray(beta_ext, float)).reshape(-1, fm.nv)
    rho_tau = float(np.max(np.abs(tau) / limits[None, :]))
    rho_vel = float(np.max(np.abs(qd_des) / vlim[None, :]))
    return {"rho_tau": rho_tau, "rho_vel": rho_vel, "rho": max(rho_tau, rho_vel)}


def iiwa_rollout_ext(fm: FrictionIiwa, beta_hat_ext: np.ndarray, q_start, q_goal,
                     cfg: IiwaL3Config) -> dict:
    """FF(beta_hat_ext)+inertia-weighted PD reach-and-hold on the friction plant."""
    nv = fm.nv
    limits = fm.base.effort_limit() * cfg.torque_scale
    vlim = np.array(fm.base.model.velocityLimit, float)
    q = np.asarray(q_start, float).copy()
    qd = np.zeros(nv)
    q_des, qd_des, qdd_des = minimum_jerk_trajectory(
        q_start, q_goal, cfg.n_steps, cfg.dt
    )
    ee_goal = fm.base.forward_kinematics(q_goal)

    # Frozen inertia weighting from the TRUE model at the start pose (fixed,
    # calibration-independent, as in the S3/S5 feedforward-plus-fixed-PD design).
    M0 = fm.base.mass_matrix(np.asarray(q_start, float)) + np.diag(fm.armature)

    ee_err = np.zeros(cfg.n_steps)
    qds = np.zeros((cfg.n_steps, nv))
    sat_hits = 0
    fault = False
    diverged = False
    for k in range(cfg.n_steps):
        tau_ff = regressor_ext_iiwa(q, qd, qdd_des[k], fm) @ beta_hat_ext
        tau_pd = np.clip(
            M0 @ (cfg.kp * (q_des[k] - q) + cfg.kd * (qd_des[k] - qd)),
            -cfg.pd_authority * limits, cfg.pd_authority * limits,
        )
        tau_sat = np.clip(tau_ff + tau_pd, -limits, limits)
        sat_hits += int(np.sum(np.abs(tau_ff + tau_pd - tau_sat) > 1e-9))
        qdd = fm.forward_dynamics(q, qd, tau_sat)
        qd = qd + qdd * cfg.dt
        q = q + qd * cfg.dt
        qds[k] = qd
        ee_err[k] = float(np.linalg.norm(fm.base.forward_kinematics(q) - ee_goal))
        if not np.all(np.isfinite(q)) or np.max(np.abs(q)) > 1e3:
            diverged = True
            ee_err[k:] = ee_err[k]
            qds[k:] = qd
            break
        if np.any(np.abs(qd) > vlim):
            fault = True
            ee_err[k:] = ee_err[k]
            qds[k:] = qd
            break

    settle = max(1, int(cfg.settle_frac * cfg.n_steps))
    ok = bool(not diverged and not fault)
    if ok:
        hold_ee = float(np.mean(ee_err[-settle:]))
        hold_vel = float(np.mean(np.linalg.norm(qds[-settle:], axis=1)))
    else:
        hold_ee = hold_vel = float("inf")
    return {
        "hold_ee_error": hold_ee, "hold_velocity": hold_vel,
        "saturation_fraction": sat_hits / (cfg.n_steps * nv),
        "fault": fault, "diverged": diverged, "finite": ok,
    }


def iiwa_failure_ext(r: dict, cfg: IiwaL3Config) -> bool:
    if not r["finite"]:
        return True
    return bool(r["hold_ee_error"] > cfg.pos_threshold
                or r["hold_velocity"] > cfg.vel_threshold)


def null_basis(V_base: np.ndarray) -> np.ndarray:
    _, _, vt = np.linalg.svd(V_base)
    return vt[V_base.shape[0]:, :].copy()


def make_beta_hat_ext_iiwa(beta_true, V_base, Nb, cal_scale, rng) -> np.ndarray:
    beta = np.asarray(beta_true, float).copy()
    if cal_scale > 0:
        a = rng.normal(size=V_base.shape[0])
        a *= cal_scale / (np.linalg.norm(a) + 1e-12)
        beta = beta + V_base.T @ a
    return beta


def run_iiwa_terminal_grid(fm: FrictionIiwa, V_base: np.ndarray, n_seeds: int) -> list[dict]:
    beta_true = fm.beta_ext_true()
    Nb = null_basis(V_base)
    q_start = np.asarray(Q_START, float)
    rows = []
    for cal_name, cal in CAL_SCALES.items():
        for tl_name, ts in TORQUE_SCALES.items():
            for hz_name, hz in HORIZONS.items():
                for tg_name, tg in TARGETS.items():
                    for seed in range(n_seeds):
                        rng = np.random.default_rng([seed, stable_seed(cal_name)])
                        beta_hat = make_beta_hat_ext_iiwa(beta_true, V_base, Nb, cal, rng)
                        cfg = IiwaL3Config(torque_scale=ts, n_steps=hz)
                        r = iiwa_rollout_ext(fm, beta_hat, q_start, tg, cfg)
                        feas = feasibility_components_iiwa(fm, beta_true, q_start, tg, cfg)
                        feas_est = feasibility_components_iiwa(fm, beta_hat, q_start, tg, cfg)
                        a_err = V_base @ (beta_hat - beta_true)
                        rows.append({
                            "cal": cal_name, "torque": tl_name, "horizon": hz_name,
                            "target": tg_name, "seed": seed,
                            "calibration_alpha_rmse": float(np.sqrt(np.mean(a_err**2))),
                            "failure": bool(iiwa_failure_ext(r, cfg)),
                            "fault": bool(r["fault"]),
                            "hold_ee_error": r["hold_ee_error"],
                            "feasibility_risk_ext": feas["rho"],
                            "rho_tau": feas["rho_tau"], "rho_vel": feas["rho_vel"],
                            "feasibility_risk_ext_est": feas_est["rho"],
                            # Torque-only ablation of the ONLINE score (see s5_layer3).
                            "feasibility_risk_tau_only": feas_est["rho_tau"],
                        })
    return rows
