"""S5-3: dynamic (torque-level) Layer-2 execution on the friction 4R plant.

The S2 Layer-2 experiments integrated the resolved-rate reference kinematically
and assumed it was followed exactly. Here the same task-compatible references
(null-space excitation over an end-effector hold task) are EXECUTED: an
inverse-dynamics feedforward (nominal extended parameters) plus PD controller
tracks the reference on the true friction plant, and identification uses
measured states and measured torque only -- the same closed-loop pipeline as
the S5 Layer-1 experiments, generalized to the n-link chain.

Forward dynamics is assembled numerically from the verified inverse-dynamics
regressor (no new symbolic derivation):

    g(q)        = ID(q, 0, 0)
    h(q, qd)    = ID(q, qd, 0)              (Coriolis + gravity)
    M(q) e_j    = ID(q, 0, e_j) - g(q)
    qdd         = M^{-1} (tau - h(q, qd) - tau_fric(qd))

The friction model matches the 2R S5 model: viscous + smoothed Coulomb per
joint, identical in the plant and in the extended regressor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .planar_nr import PlanarNRModel, forward_kinematics, inverse_dynamics, regressor
from .posterior import GaussianPosterior
from .s5_execution import estimate_qdd
from .tasks import Task
from .task_policies import NullspaceConfig, rollout


@dataclass(frozen=True)
class FrictionModelNR:
    """n-link planar model with viscous + smoothed-Coulomb joint friction."""

    rigid: PlanarNRModel
    fv: tuple[float, ...]
    fc: tuple[float, ...]
    eps: float = 0.05

    @property
    def n(self) -> int:
        return self.rigid.n_links

    def beta_ext_true(self) -> np.ndarray:
        fr = np.empty(2 * self.n)
        fr[0::2] = self.fv
        fr[1::2] = self.fc
        return np.concatenate([self.rigid.beta_true(), fr])

    def friction_torque(self, qd: np.ndarray) -> np.ndarray:
        qd = np.asarray(qd, float)
        return np.asarray(self.fv) * qd + np.asarray(self.fc) * np.tanh(qd / self.eps)


def default_friction_nr(model: PlanarNRModel) -> FrictionModelNR:
    """Per-joint coefficients tapering toward the distal joints (like the 2R set)."""
    n = model.n_links
    fv = tuple(0.40 * (0.8**i) for i in range(n))
    fc = tuple(0.80 * (0.8**i) for i in range(n))
    return FrictionModelNR(rigid=model, fv=fv, fc=fc)


def friction_regressor_nr(qd: np.ndarray, eps: float) -> np.ndarray:
    """(n x 2n) friction block: row i has [qd_i, tanh(qd_i/eps)] in its pair."""
    qd = np.asarray(qd, float)
    n = qd.size
    F = np.zeros((n, 2 * n))
    for i in range(n):
        F[i, 2 * i] = qd[i]
        F[i, 2 * i + 1] = np.tanh(qd[i] / eps)
    return F


def regressor_ext_nr(q, qd, qdd, fm: FrictionModelNR) -> np.ndarray:
    Y = regressor(q, qd, qdd, fm.rigid)
    return np.hstack([Y, friction_regressor_nr(qd, fm.eps)])


def stacked_regressor_ext_nr(samples: np.ndarray, fm: FrictionModelNR) -> np.ndarray:
    n = fm.n
    rows = [
        regressor_ext_nr(s[0:n], s[n:2 * n], s[2 * n:3 * n], fm) for s in samples
    ]
    return np.vstack(rows)


def forward_dynamics_nr(q, qd, tau, fm: FrictionModelNR) -> np.ndarray:
    """qdd = M^{-1} (tau - h(q,qd) - tau_fric), M assembled from inverse dynamics."""
    n = fm.n
    q = np.asarray(q, float)
    qd = np.asarray(qd, float)
    g_vec = inverse_dynamics(q, np.zeros(n), np.zeros(n), fm.rigid)
    h_vec = inverse_dynamics(q, qd, np.zeros(n), fm.rigid)
    M = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        M[:, j] = inverse_dynamics(q, np.zeros(n), e, fm.rigid) - g_vec
    rhs = np.asarray(tau, float) - h_vec - fm.friction_torque(qd)
    return np.linalg.solve(M, rhs)


# ---------------------------------------------------------------------------
# Closed-loop execution (n-link version of s5_execution.execute_reference)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionConfigNR:
    # Per-unit-inertia PD gains, scaled by the NOMINAL mass-matrix diagonal at
    # the start pose (omega = sqrt(kp) = 10 rad/s, critically damped). Scalar
    # gains are unstable here: the 4R joint inertias span two orders of
    # magnitude, so a gain that stabilizes joint 1 over-drives joint 4. The
    # 2e-3 control period (500 Hz) is required for discrete stability of the
    # coupled loop (the kinematic S2 rollouts ran at 1e-2, which is too coarse
    # for torque-level tracking).
    kp: float = 100.0
    kd: float = 20.0
    pd_authority: float = 0.3  # PD torque clipped to this fraction of the limits;
    #                            bounded feedback authority (as on industrial
    #                            torque-controlled arms) prevents a saturation
    #                            cascade when the nominal feedforward is off
    # Per-joint effort limits, sized like a real arm datasheet (~2x the peak
    # gravity load per joint: 58/29/13/2 N m at full stretch). A uniform limit
    # equal to the proximal gravity load would make free excitation
    # structurally infeasible on this arm.
    torque_limit: tuple[float, ...] = (120.0, 80.0, 40.0, 20.0)
    dt: float = 2e-3
    sigma_q: float = 1e-4
    sigma_qd: float = 1e-3
    sigma_tau: float = 1e-2
    q_fault: float = 1.15 * np.pi
    qd_fault: float = 3.5
    smooth_window: int = 21
    obs_decimate: int = 5
    edge_trim: int = 21
    qd_min: float = 0.05  # velocity row filtering (see s5_execution)


def execute_reference_nr(
    fm: FrictionModelNR,
    q_des: np.ndarray, qd_des: np.ndarray, qdd_des: np.ndarray,
    beta_ctrl: np.ndarray,
    cfg: ExecutionConfigNR,
    rng: np.random.Generator,
) -> dict:
    n = fm.n
    n_steps = q_des.shape[0]
    q = q_des[0].copy()
    qd = qd_des[0].copy()
    # Inertia-weighted PD: tau_pd = M0 (kp e + kd ed) with the FULL nominal
    # mass matrix at the start pose (assembled by the inverse-dynamics column
    # trick on the nominal rigid parameters, then frozen -- it is part of the
    # fixed feedback, not the calibrated estimate). Diagonal per-joint gains
    # are discretely UNSTABLE on this arm: the coupled mode
    # eig(M^-1 diag(kd)) reaches ~kd_1/lambda_min(M), which at any usable
    # proximal gain exceeds the 2/dt bound and produces a sample-rate buzz.
    # With the full M0, M^-1 M0 ~ I and the loop decouples.
    beta_rigid = np.asarray(beta_ctrl, float)[: 3 * n]
    Y0g = regressor(q_des[0], np.zeros(n), np.zeros(n), fm.rigid)
    g0 = Y0g @ beta_rigid
    M0 = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        M0[:, j] = regressor(q_des[0], np.zeros(n), e, fm.rigid) @ beta_rigid - g0
    M0 = 0.5 * (M0 + M0.T)
    q_true = np.zeros((n_steps, n))
    qd_true = np.zeros((n_steps, n))
    q_meas = np.zeros((n_steps, n))
    qd_meas = np.zeros((n_steps, n))
    tau_meas = np.zeros((n_steps, n))
    sat_hits = 0
    fault = False
    for k in range(n_steps):
        qm = q + rng.normal(0.0, cfg.sigma_q, n)
        qdm = qd + rng.normal(0.0, cfg.sigma_qd, n)
        tau_ff = regressor_ext_nr(qm, qdm, qdd_des[k], fm) @ beta_ctrl
        limits = np.asarray(cfg.torque_limit, float)
        tau_pd = np.clip(
            M0 @ (cfg.kp * (q_des[k] - qm) + cfg.kd * (qd_des[k] - qdm)),
            -cfg.pd_authority * limits, cfg.pd_authority * limits,
        )
        tau = tau_ff + tau_pd
        tau_sat = np.clip(tau, -limits, limits)
        sat_hits += int(np.sum(np.abs(tau - tau_sat) > 1e-9))
        qdd = forward_dynamics_nr(q, qd, tau_sat, fm)
        qd = qd + qdd * cfg.dt
        q = q + qd * cfg.dt
        q_true[k] = q
        qd_true[k] = qd
        q_meas[k] = qm
        qd_meas[k] = qdm
        tau_meas[k] = tau_sat + rng.normal(0.0, cfg.sigma_tau, n)
        if np.max(np.abs(q)) > cfg.q_fault or np.max(np.abs(qd)) > cfg.qd_fault:
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
        "saturation_fraction": sat_hits / max(1, q_true.shape[0] * n),
        "tracking_rmse": float(np.sqrt(np.mean(track**2))),
        "n_steps": q_true.shape[0],
    }


def build_observations_nr(
    execu: dict, V_base: np.ndarray, fm: FrictionModelNR, cfg: ExecutionConfigNR
) -> tuple[np.ndarray, np.ndarray]:
    qdd_est = estimate_qdd(execu["qd_meas"], cfg.dt, cfg.smooth_window)
    states = np.hstack([execu["q_meas"], execu["qd_meas"], qdd_est])
    t0, t1 = cfg.edge_trim, states.shape[0] - cfg.edge_trim
    keep = np.arange(t0, t1, cfg.obs_decimate)
    states = states[keep]
    W_base = stacked_regressor_ext_nr(states, fm) @ V_base.T
    y = execu["tau_meas"][keep].reshape(-1)
    n = fm.n
    # Velocity row filtering (see s5_execution.build_observations).
    row_qd = np.abs(states[:, n:2 * n]).reshape(-1)
    mask = row_qd >= cfg.qd_min
    return W_base[mask], y[mask]


# ---------------------------------------------------------------------------
# Layer-2 experiment: task-compatible vs free excitation, executed dynamically
# ---------------------------------------------------------------------------

S5_LAYER2_POLICIES = (
    "free_reference",            # executable free excitation (upper bound)
    "no_exploration",            # pure task
    "smooth_random_nullspace",   # task + smooth random null-space excitation
    "fourier_nullspace",         # task + Fourier null-space excitation
)


@dataclass(frozen=True)
class S5Layer2Settings:
    prior_std: float = 0.15  # honest CAD-error-scale prior (see s5_layer1)
    param_error: float = 0.2
    friction_error: float = 0.5
    target_sum_sq_qdd: float = 5000.0
    holdout_n: int = 1200
    holdout_seed: int = 9999


def nominal_beta_ext_nr(fm: FrictionModelNR, settings: S5Layer2Settings,
                        rng: np.random.Generator) -> np.ndarray:
    """CAD-like nominal extended parameters (mirrors the 2R construction)."""
    r = fm.rigid
    e = settings.param_error
    rigid = []
    for i in range(fm.n):
        m = r.masses[i] * (1 + rng.uniform(-e, e))
        rr = r.com_dists[i] * (1 + rng.uniform(-e, e))
        ii = r.inertias[i] * (1 + rng.uniform(-e, e))
        rigid += [m, m * rr, ii + m * rr**2]
    ef = settings.friction_error
    fr = np.empty(2 * fm.n)
    fr[0::2] = np.asarray(fm.fv) * (1 + rng.uniform(-ef, ef, fm.n))
    fr[1::2] = np.asarray(fm.fc) * (1 + rng.uniform(-ef, ef, fm.n))
    return np.concatenate([np.asarray(rigid), fr])


def _kinematic_reference(policy: str, fm: FrictionModelNR, task: Task,
                         ns_cfg: NullspaceConfig, seed: int, dt: float):
    """Task-compatible reference from the resolved-rate rollout (design-time)."""
    res = rollout(fm.rigid, task, policy, ns_cfg, seed=seed)
    n = fm.n
    states = res["states"]
    q, qd = states[:, :n], states[:, n:2 * n]
    qdd = np.gradient(qd, dt, axis=0)
    return q, qd, qdd


def _free_reference(fm: FrictionModelNR, n_steps: int, dt: float, q0: np.ndarray,
                    rng: np.random.Generator, vel_limit: float, acc_limit: float,
                    beta_nominal: np.ndarray, tau_limit: float,
                    torque_margin: float = 0.6, q_range: float = np.pi):
    # margin 0.6: the probe uses the NOMINAL parameters, so the true required
    # torque can exceed it by the (unknown) parameter error plus feedback
    # action; 40% headroom keeps the executed motion out of saturation.
    """Executable free excitation: box- and torque-fitted periodic Fourier.

    The n-joint analogue of the Layer-1 fourier_envelope policy: a periodic
    finite-Fourier shape whose amplitude saturates the position range and
    whose time dilation saturates the tighter of the velocity/acceleration
    bounds, entered via a quintic transition from the start pose at the
    nearest-position phase. Unlike the planar 2R, the 4R's gravity torque at
    full stretch nearly equals the actuator limit, so the amplitude is also
    shrunk until the NOMINAL-model torque stays inside the budget -- free
    excitation on this arm cannot legally sweep the whole workspace.
    """
    n = fm.n
    K = 5
    s_dense = np.linspace(0.0, 2.0, 400)
    a = rng.normal(0.0, 1.0, size=(K, n))
    b = rng.normal(0.0, 1.0, size=(K, n))
    q_u = np.zeros((s_dense.size, n))
    qd_u = np.zeros_like(q_u)
    qdd_u = np.zeros_like(q_u)
    for k in range(1, K + 1):
        w = 2.0 * np.pi * k
        sk = np.sin(w * s_dense)[:, None]
        ck = np.cos(w * s_dense)[:, None]
        q_u += (a[k - 1] * sk + b[k - 1] * ck) / k
        qd_u += (a[k - 1] * w * ck - b[k - 1] * w * sk) / k
        qdd_u += (-a[k - 1] * w**2 * sk - b[k - 1] * w**2 * ck) / k
    # Center the oscillation at the start pose: a zero-mean Fourier shape
    # shrunk toward q = 0 would park this arm at the fully stretched
    # horizontal pose, where gravity alone nearly saturates the actuators.
    q_c = np.asarray(q0, float)
    amp = max(1e-3, (q_range - float(np.max(np.abs(q_c)))) / max(np.max(np.abs(q_u)), 1e-12))

    limits = np.asarray(tau_limit, float) * torque_margin

    def _torque_ok(amp_, lam_):
        probe = np.hstack([
            q_c + amp_ * q_u, amp_ * lam_ * qd_u, amp_ * lam_**2 * qdd_u
        ])[::4]
        tau = (stacked_regressor_ext_nr(probe, fm)
               @ np.asarray(beta_nominal, float)).reshape(-1, n)
        return bool(np.all(np.max(np.abs(tau), axis=0) <= limits))

    # Fit the time dilation to the velocity/acceleration box once, then SLOW
    # DOWN (shrink lam) to meet the torque budget; shrinking the amplitude
    # instead would backfire, because refitting lam to the velocity bound at
    # smaller amplitude raises the accelerations (same speed over smaller
    # motions). Amplitude is only reduced if slowing down cannot help (i.e.
    # gravity alone violates the budget somewhere in the swept workspace).
    # Bandwidth cap: the top harmonic (K * lam Hz) must stay well inside the
    # acceleration-estimation filter passband, or the reconstructed qdd
    # misses real signal and the residuals blow up. 0.8 keeps the top
    # harmonic at 4 Hz (the 21-sample / 42 ms smoother passes < ~8 Hz).
    lam_cap = 0.8
    for _ in range(8):
        lam_v = vel_limit / max(amp * np.max(np.abs(qd_u)), 1e-12)
        lam_a = float(np.sqrt(acc_limit / max(amp * np.max(np.abs(qdd_u)), 1e-12)))
        lam = min(lam_v, lam_a, lam_cap)
        ok = False
        for _ in range(20):
            if _torque_ok(amp, lam):
                ok = True
                break
            lam *= 0.9
        if ok:
            break
        amp *= 0.8

    def entry(s0):
        qe = q_c + amp * np.array([np.interp(s0, s_dense, q_u[:, j]) for j in range(n)])
        qde = amp * lam * np.array([np.interp(s0, s_dense, qd_u[:, j]) for j in range(n)])
        return qe, qde

    s_grid = np.linspace(0.0, 2.0, 64, endpoint=False)
    s0 = float(s_grid[int(np.argmin([
        np.max(np.abs(entry(s)[0] - q0)) for s in s_grid
    ]))])
    q_e, qd_e = entry(s0)

    from .s5_trajectories import quintic_segment
    n_trans = max(2, int(round(0.8 / dt)))  # 0.8 s transition
    for _ in range(6):
        qt, qdt, qddt = quintic_segment(q0, np.zeros(n), q_e, qd_e, n_trans, dt)
        if (np.max(np.abs(qdt)) <= vel_limit and np.max(np.abs(qddt)) <= acc_limit
                and np.max(np.abs(qt)) <= q_range):
            break
        n_trans = int(n_trans * 1.5)
    n_rest = max(1, n_steps - qt.shape[0])
    t = np.arange(1, n_rest + 1) * dt
    s = (s0 + lam * t) % 2.0
    qf = q_c + amp * np.vstack([np.interp(s, s_dense, q_u[:, j]) for j in range(n)]).T
    qdf = amp * lam * np.vstack([np.interp(s, s_dense, qd_u[:, j]) for j in range(n)]).T
    qddf = amp * lam**2 * np.vstack([np.interp(s, s_dense, qdd_u[:, j]) for j in range(n)]).T
    return (np.vstack([qt, qf]), np.vstack([qdt, qdf]), np.vstack([qddt, qddf]))


def _calibrate_amp(policy, fm, task, ns_cfg, settings, dt, seed=0):
    """Scale null-space amplitude so the reference qdd energy hits the target."""
    if policy == "no_exploration":
        return ns_cfg
    _, _, qdd = _kinematic_reference(policy, fm, task, ns_cfg, seed, dt)
    e0 = float(np.sum(qdd**2))
    if e0 <= 0:
        return ns_cfg
    import dataclasses as _dc
    return _dc.replace(ns_cfg, amp=ns_cfg.amp * float(np.sqrt(settings.target_sum_sq_qdd / e0)))


def run_layer2_closed_loop(
    fm: FrictionModelNR,
    V_base: np.ndarray,
    task: Task,
    settings: S5Layer2Settings,
    exec_cfg: ExecutionConfigNR,
    n_seeds: int,
    policies: tuple[str, ...] = S5_LAYER2_POLICIES,
) -> dict:
    alpha_true = V_base @ fm.beta_ext_true()
    rng_h = np.random.default_rng(settings.holdout_seed)
    n = fm.n
    hold_states = np.hstack([
        rng_h.uniform(-np.pi, np.pi, (settings.holdout_n, n)),
        rng_h.uniform(-2.0, 2.0, (settings.holdout_n, n)),
        rng_h.uniform(-4.0, 4.0, (settings.holdout_n, n)),
    ])
    W_holdout = stacked_regressor_ext_nr(hold_states, fm) @ V_base.T

    # smooth_window scales with the control period: 75 steps at 2e-3 = 150 ms,
    # matching the spectral content the S2 profiles had at their 10 ms period.
    # An unscaled window leaves gradient-amplified high-frequency qdd that the
    # energy calibration then "pays for", collapsing the actual motion.
    base_ns = NullspaceConfig(smooth_window=75)
    amps = {
        p: _calibrate_amp(p, fm, task, base_ns, settings, exec_cfg.dt)
        for p in policies if p not in ("free_reference",)
    }

    by_seed = []
    for seed in range(n_seeds):
        rng_nom = np.random.default_rng([seed, 5])
        beta_ctrl = nominal_beta_ext_nr(fm, settings, rng_nom)
        for policy in policies:
            if policy == "free_reference":
                ref = _free_reference(
                    fm, task.n_steps, exec_cfg.dt,
                    np.asarray(task.q_init, float),
                    np.random.default_rng([seed, 21]),
                    vel_limit=2.0, acc_limit=4.0,
                    beta_nominal=beta_ctrl, tau_limit=exec_cfg.torque_limit,
                )
            else:
                ref = _kinematic_reference(policy, fm, task, amps[policy], seed, exec_cfg.dt)
            execu = execute_reference_nr(
                fm, *ref, beta_ctrl, exec_cfg, np.random.default_rng([seed, 3])
            )
            W_base, y = build_observations_nr(execu, V_base, fm, exec_cfg)
            a_prior = V_base @ beta_ctrl
            post = GaussianPosterior.isotropic_prior(a_prior, settings.prior_std)
            if y.size:
                pass1 = GaussianPosterior.isotropic_prior(a_prior, settings.prior_std)
                pass1.update(W_base, y, exec_cfg.sigma_tau)
                resid = y - W_base @ pass1.mean()
                sigma_eff = float(max(exec_cfg.sigma_tau, np.sqrt(np.mean(resid**2))))
                post.update(W_base, y, sigma_eff)
            else:  # all rows velocity-filtered: stay at the prior
                sigma_eff = exec_cfg.sigma_tau
            mu = post.mean()

            # Task error from the EXECUTED motion (was: kinematic assumption).
            p0 = forward_kinematics(np.asarray(task.q_init, float), fm.rigid)
            ee = np.array([
                np.linalg.norm(forward_kinematics(qk, fm.rigid) - p0)
                for qk in execu["q_true"][:: max(1, execu["q_true"].shape[0] // 100)]
            ])
            by_seed.append({
                "policy": policy, "seed": seed,
                "logdet_cov": post.logdet_cov(),
                "alpha_rmse": float(np.sqrt(np.mean((mu - alpha_true) ** 2))),
                "holdout_torque_rmse": float(np.sqrt(np.mean(
                    (W_holdout @ (mu - alpha_true)) ** 2))),
                "sigma_eff": sigma_eff,
                "ee_error_mean": float(np.mean(ee)) if policy != "free_reference" else float("nan"),
                "ee_error_max": float(np.max(ee)) if policy != "free_reference" else float("nan"),
                "tracking_rmse": execu["tracking_rmse"],
                "fault": bool(execu["fault"]),
                "max_abs_qd": float(np.max(np.abs(execu["qd_true"]))),
                "peak_abs_tau": float(np.max(np.abs(execu["tau_meas"]))),
                "sum_sq_qdd_ref": float(np.sum(ref[2] ** 2)),
            })

    keys = ["logdet_cov", "alpha_rmse", "holdout_torque_rmse", "sigma_eff",
            "ee_error_mean", "ee_error_max", "tracking_rmse", "max_abs_qd",
            "peak_abs_tau", "sum_sq_qdd_ref"]

    def _agg(vals):
        arr = np.array([v for v in vals if np.isfinite(v)], float)
        if arr.size == 0:
            return {"mean": float("nan"), "std": float("nan")}
        return {"mean": float(arr.mean()), "std": float(arr.std())}

    summary = {}
    for p in policies:
        rows = [r for r in by_seed if r["policy"] == p]
        summary[p] = {
            "n_seeds": len(rows),
            "n_faults": int(sum(r["fault"] for r in rows)),
            **{k: _agg([r[k] for r in rows]) for k in keys},
        }
    return {"by_seed": by_seed, "summary": summary}


def gap_fractions_ext(summary: dict) -> dict:
    """Fraction of the free-reference improvement over no_exploration recovered."""
    out = {}
    for metric in ("logdet_cov", "holdout_torque_rmse"):
        ref = summary["free_reference"][metric]["mean"]
        base = summary["no_exploration"][metric]["mean"]
        span = base - ref
        out[metric] = {
            p: (float((base - summary[p][metric]["mean"]) / span)
                if abs(span) > 1e-12 else float("nan"))
            for p in summary if p not in ("free_reference", "no_exploration")
        }
    return out
