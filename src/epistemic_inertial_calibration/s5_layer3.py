"""S5-4: terminal feasibility on the friction plant with fault-consistent labels.

Rebuilds the Layer-3 experiment in the S5 world so that both the risk score
and the failure labels are hardware-consistent:

1. The plant has friction; the controller feedforward uses the extended
   estimate (rigid + friction parameters).
2. The feasibility risk has BOTH a torque component and a velocity component,

       rho_tau = max_t max_i |Y_ext(q_d, qd_d, qdd_d) beta|_i / tau_max_i
       rho_vel = max_t max_i |qd_{d,i}(t)| / qd_max_i
       rho     = max(rho_tau, rho_vel),

   evaluated on the DESIRED trajectory. With beta = beta_true (held fixed across the
   sweep) this is the evaluator's parameter-error-independent reference score; with
   beta = beta_hat it is the score a robot can compute online from its own estimate. A torque-only score is blind to
   velocity-limit infeasibility, which on real hardware trips a controller
   fault before torque saturation ever matters.
3. The rollout enforces the same limits the score checks: exceeding the
   velocity bound (or diverging) faults the run, exactly as an industrial
   controller would E-stop. A faulted run is a failed run. Torque saturates
   at the limit as before.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controllers import reach_hold_trajectory
from .evaluation import stable_seed
from .s5_model import (
    FrictionModel2R,
    S5ProjectionArtifact,
    forward_dynamics_ext,
    regressor_ext,
    stacked_regressor_ext,
)


# ---------------------------------------------------------------------------
# Controller config and closed-loop rollout (friction plant, fault monitor)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class S5ControllerConfig:
    kp: float = 60.0
    kd: float = 15.0
    torque_limit: float = 40.0
    qd_limit: float = 4.0        # actuator velocity rating (rad/s); fault above
    dt: float = 2e-3
    n_steps: int = 750           # 1.5 s horizon
    settle_frac: float = 0.2
    pos_threshold: float = 0.05
    vel_threshold: float = 0.05


def closed_loop_rollout_ext(
    fmodel: FrictionModel2R,
    beta_hat_ext: np.ndarray,
    q_start,
    q_goal,
    cfg: S5ControllerConfig,
) -> dict:
    """FF(beta_hat_ext) + PD reach-and-hold on the friction plant with faults.

    The velocity fault emulates a hardware controller: exceeding qd_limit
    aborts the run (fault=True), which counts as task failure.
    """
    q = np.asarray(q_start, float).copy()
    qd = np.zeros(2)
    q_des, qd_des, qdd_des = reach_hold_trajectory(q_start, q_goal, cfg.n_steps, cfg.dt)

    qs = np.zeros((cfg.n_steps, 2))
    qds = np.zeros((cfg.n_steps, 2))
    sat_hits = 0
    fault = False
    diverged = False
    for k in range(cfg.n_steps):
        tau_ff = regressor_ext(q, qd, qdd_des[k], fmodel) @ beta_hat_ext
        tau = tau_ff + cfg.kp * (q_des[k] - q) + cfg.kd * (qd_des[k] - qd)
        tau_sat = np.clip(tau, -cfg.torque_limit, cfg.torque_limit)
        sat_hits += int(np.sum(np.abs(tau - tau_sat) > 1e-9))
        qdd = forward_dynamics_ext(q, qd, tau_sat, fmodel)
        qd = qd + qdd * cfg.dt
        q = q + qd * cfg.dt
        qs[k] = q
        qds[k] = qd
        if not np.all(np.isfinite(q)) or np.max(np.abs(q)) > 1e3:
            diverged = True
            qs[k:] = q
            qds[k:] = qd
            break
        if np.max(np.abs(qd)) > cfg.qd_limit:
            fault = True
            qs[k:] = q
            qds[k:] = qd
            break

    settle = max(1, int(cfg.settle_frac * cfg.n_steps))
    q_goal = np.asarray(q_goal, float)
    ok = bool(not diverged and not fault and np.all(np.isfinite(qs)))
    if ok:
        hold_err = float(np.mean(np.linalg.norm(qs[-settle:] - q_goal, axis=1)))
        hold_vel = float(np.mean(np.linalg.norm(qds[-settle:], axis=1)))
    else:
        hold_err = hold_vel = float("inf")
    return {
        "hold_error": hold_err,
        "hold_velocity": hold_vel,
        "saturation_fraction": sat_hits / (cfg.n_steps * 2),
        "fault": fault,
        "diverged": diverged,
        "finite": ok,
    }


def task_failure_ext(rollout: dict, cfg: S5ControllerConfig) -> bool:
    if not rollout["finite"]:
        return True  # fault or divergence is a failure
    return bool(
        rollout["hold_error"] > cfg.pos_threshold
        or rollout["hold_velocity"] > cfg.vel_threshold
    )


# ---------------------------------------------------------------------------
# Extended feasibility risk (torque + velocity components)
# ---------------------------------------------------------------------------


def feasibility_components(
    fmodel: FrictionModel2R, beta_ext: np.ndarray, q_start, q_goal,
    cfg: S5ControllerConfig, stride: int = 5,
) -> dict:
    """rho_tau, rho_vel, and their max on the desired reach-and-hold trajectory.

    beta_ext = extended nominal parameters for the evaluator score, or the
    extended estimate for the online score. Velocity needs no parameters.
    """
    q_des, qd_des, qdd_des = reach_hold_trajectory(q_start, q_goal, cfg.n_steps, cfg.dt)
    states = np.hstack([q_des, qd_des, qdd_des])[::stride]
    tau = (stacked_regressor_ext(states, fmodel) @ np.asarray(beta_ext, float)).reshape(-1, 2)
    rho_tau = float(np.max(np.abs(tau)) / cfg.torque_limit)
    rho_vel = float(np.max(np.abs(qd_des)) / cfg.qd_limit)
    return {"rho_tau": rho_tau, "rho_vel": rho_vel, "rho": max(rho_tau, rho_vel)}


# ---------------------------------------------------------------------------
# Calibration-error injection in the extended base coordinates
# ---------------------------------------------------------------------------


def null_basis(V_base: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the non-identifiable null space (rows x 10)."""
    _, _, vt = np.linalg.svd(V_base)
    return vt[V_base.shape[0]:, :].copy()


def make_beta_hat_ext(
    beta_true_ext: np.ndarray, V_base: np.ndarray, Nb: np.ndarray,
    cal_scale: float, pin_scale: float, rng: np.random.Generator,
) -> np.ndarray:
    """beta_hat = truth + base-coordinate error + optional null-space pin shift."""
    beta = np.asarray(beta_true_ext, float).copy()
    if cal_scale > 0:
        a = rng.normal(size=V_base.shape[0])
        a *= cal_scale / (np.linalg.norm(a) + 1e-12)
        beta = beta + V_base.T @ a
    if pin_scale > 0 and Nb.shape[0] > 0:
        p = rng.normal(size=Nb.shape[0])
        p *= pin_scale / (np.linalg.norm(p) + 1e-12)
        beta = beta + Nb.T @ p
    return beta


def calibration_alpha_rmse_ext(beta_hat, beta_true, V_base) -> float:
    a = V_base @ (np.asarray(beta_hat, float) - np.asarray(beta_true, float))
    return float(np.sqrt(np.mean(a**2)))


# ---------------------------------------------------------------------------
# Sweep grid (mirrors S3-A/B structure on the extended model)
# ---------------------------------------------------------------------------

CAL_SCALES = {"cal0": 0.0, "cal_small": 0.05, "cal_medium": 0.2, "cal_large": 0.6}
TORQUE_LIMITS = {"loose": 60.0, "medium": 12.0, "tight": 6.0}
# "mid" horizons and the "wide" target create VELOCITY-only infeasible
# conditions (rho_vel > 1 with rho_tau < 1): on hardware the velocity rating
# often binds before torque, and a torque-only score is blind to it.
HORIZONS = {"normal": 750, "mid": 500, "short": 140}
TARGETS = {"near": (0.9, -0.6), "far": (1.6, -1.3), "wide": (2.6, -2.3)}
Q_START = (0.2, 0.3)


def _holdout_regressor_ext(fmodel: FrictionModel2R, n: int = 1500, seed: int = 9991):
    rng = np.random.default_rng(seed)
    states = np.hstack([
        rng.uniform(-np.pi, np.pi, size=(n, 2)),
        rng.uniform(-2.0, 2.0, size=(n, 2)),
        rng.uniform(-4.0, 4.0, size=(n, 2)),
    ])
    return stacked_regressor_ext(states, fmodel)


def run_grid_ext(proj: S5ProjectionArtifact, n_seeds: int = 5,
                 include_pin_shift: bool = True) -> dict:
    fmodel = proj.model()
    V_base = proj.V_base
    beta_true = fmodel.beta_ext_true()
    Nb = null_basis(V_base)
    W_holdout = _holdout_regressor_ext(fmodel)

    rows = []
    for cal_name, cal in CAL_SCALES.items():
        for tl_name, tl in TORQUE_LIMITS.items():
            for hz_name, hz in HORIZONS.items():
                for tg_name, tg in TARGETS.items():
                    for seed in range(n_seeds):
                        rng = np.random.default_rng([seed, stable_seed(cal_name)])
                        beta_hat = make_beta_hat_ext(beta_true, V_base, Nb, cal, 0.0, rng)
                        rows.append(_eval_ext(
                            fmodel, beta_true, V_base, W_holdout, beta_hat,
                            cal_name, tl_name, tl, hz_name, hz, tg_name, tg, seed,
                            pin=False,
                        ))
    if include_pin_shift:
        for tl_name, tl in TORQUE_LIMITS.items():
            for seed in range(n_seeds):
                rng = np.random.default_rng([seed, 4242])
                beta_hat = make_beta_hat_ext(beta_true, V_base, Nb, 0.0, 0.6, rng)
                rows.append(_eval_ext(
                    fmodel, beta_true, V_base, W_holdout, beta_hat,
                    "cal0_pinshift", tl_name, tl, "normal", 750, "near",
                    TARGETS["near"], seed, pin=True,
                ))
    return {"rows": rows, "rank": int(proj.rank)}


def _eval_ext(fmodel, beta_true, V_base, W_holdout, beta_hat, cal_name, tl_name, tl,
              hz_name, hz, tg_name, tg, seed, pin) -> dict:
    cfg = S5ControllerConfig(torque_limit=tl, n_steps=hz)
    r = closed_loop_rollout_ext(fmodel, beta_hat, Q_START, tg, cfg)
    fail = task_failure_ext(r, cfg)
    feas_nom = feasibility_components(fmodel, beta_true, Q_START, tg, cfg)
    feas_est = feasibility_components(fmodel, beta_hat, Q_START, tg, cfg)
    resid = W_holdout @ (np.asarray(beta_hat) - np.asarray(beta_true))
    return {
        "cal": cal_name, "torque": tl_name, "horizon": hz_name, "target": tg_name,
        "seed": seed, "pin_shift": pin,
        "calibration_alpha_rmse": calibration_alpha_rmse_ext(beta_hat, beta_true, V_base),
        "holdout_torque_rmse": float(np.sqrt(np.mean(resid**2))),
        "hold_error": r["hold_error"], "hold_velocity": r["hold_velocity"],
        "saturation_fraction": r["saturation_fraction"],
        "fault": bool(r["fault"]), "diverged": bool(r["diverged"]),
        "failure": bool(fail),
        "feasibility_risk_ext": feas_nom["rho"],
        "rho_tau": feas_nom["rho_tau"],
        "rho_vel": feas_nom["rho_vel"],
        "feasibility_risk_ext_est": feas_est["rho"],
        "rho_tau_est": feas_est["rho_tau"],
        # Torque-only ablation of the ONLINE score (estimated parameters), so
        # the ablation isolates the score's form under the same information a
        # deployed robot has; the oracle rho_tau is kept above for reference.
        "feasibility_risk_tau_only": feas_est["rho_tau"],
    }
