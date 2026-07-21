"""S5-2: closed-loop Layer-1 comparison on the friction-extended 2R plant.

Per (policy, seed):
  1. Draw a NOMINAL model (true physical parameters perturbed by realistic
     errors) -- shared by all policies at the same seed for paired comparison.
     The nominal model is everything the robot knows a priori: it drives the
     feedforward controller, the design-time envelope/torque checks, the
     greedy scoring, and the posterior prior mean.
  2. Design the policy's executable reference trajectory (s5_trajectories).
  3. Execute it closed-loop on the true friction plant (s5_execution),
     logging measured states and measured torque only.
  4. Identify the extended base parameters from the measured data.
  5. Evaluate: ground-truth metrics (evaluator only) plus the
     hardware-computable validation metric (measured-torque prediction on a
     held-out executed validation trajectory).

segment_active_ig interleaves design and execution: after each executed
segment the posterior is updated from that segment's measured data, and the
next segment is selected by information gain against the CURRENT posterior.
This is the genuine closed-loop active-calibration loop.
"""

from __future__ import annotations

import dataclasses
import zlib
from dataclasses import dataclass

import numpy as np

from .posterior import GaussianPosterior
from .evaluation import alpha_rmse
from .s5_execution import ExecutionConfig, build_observations, execute_reference
from .s5_model import (
    FrictionModel2R,
    S5ProjectionArtifact,
    alpha_true_ext,
    stacked_regressor_ext,
)
from .s5_trajectories import (
    SEGMENT_POLICIES,
    DesignContext,
    S5TrajConfig,
    design_next_segment,
    design_reference,
)


@dataclass(frozen=True)
class S5Layer1Settings:
    # Honest prior: the precision matches the CAD-error scale actually drawn
    # for the nominal model (~0.15 in base coordinates). An artificially loose
    # prior lets residual model bias overwrite a good nominal model in weakly
    # excited directions (observed on the 7-DoF system).
    prior_std: float = 0.15
    param_error: float = 0.2      # relative error scale of nominal m, r, I
    friction_error: float = 0.5   # relative error scale of nominal Fv, Fc
    holdout_n: int = 2000
    holdout_seed: int = 9999
    valid_design_seed: int = 7777  # fixed validation-trajectory design stream


def stable_seed(name: str) -> int:
    return zlib.crc32(name.encode("utf-8")) % 9973


def nominal_beta_ext(fmodel: FrictionModel2R, settings: S5Layer1Settings,
                     rng: np.random.Generator) -> np.ndarray:
    """The robot's a-priori (CAD-like) extended parameter vector.

    Relative errors on the physical parameters (m, r, I) and larger relative
    errors on the friction coefficients, mapped through the barycentric
    parameterization. This is what the controller/designer knows.
    """
    r = fmodel.rigid
    e = settings.param_error
    m1 = r.m1 * (1 + rng.uniform(-e, e))
    r1 = r.r1 * (1 + rng.uniform(-e, e))
    i1 = r.I1 * (1 + rng.uniform(-e, e))
    m2 = r.m2 * (1 + rng.uniform(-e, e))
    r2 = r.r2 * (1 + rng.uniform(-e, e))
    i2 = r.I2 * (1 + rng.uniform(-e, e))
    ef = settings.friction_error
    fr = np.array([fmodel.fv[0], fmodel.fc[0], fmodel.fv[1], fmodel.fc[1]])
    fr = fr * (1 + rng.uniform(-ef, ef, size=4))
    rigid = np.array([
        m1, m1 * r1, i1 + m1 * r1**2,
        m2, m2 * r2, i2 + m2 * r2**2,
    ])
    return np.concatenate([rigid, fr])


def build_holdout(proj: S5ProjectionArtifact, fmodel: FrictionModel2R,
                  settings: S5Layer1Settings) -> np.ndarray:
    """Fixed noise-free holdout base regressor (evaluator only)."""
    rng = np.random.default_rng(settings.holdout_seed)
    n = settings.holdout_n
    states = np.hstack([
        rng.uniform(-np.pi, np.pi, (n, 2)),
        rng.uniform(-2.0, 2.0, (n, 2)),
        rng.uniform(-4.0, 4.0, (n, 2)),
    ])
    return stacked_regressor_ext(states, fmodel) @ proj.V_base.T


def execute_validation(
    proj: S5ProjectionArtifact, fmodel: FrictionModel2R, beta_ctrl: np.ndarray,
    traj_cfg: S5TrajConfig, exec_cfg: ExecutionConfig,
    settings: S5Layer1Settings, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Design + execute the held-out validation trajectory for this seed.

    The design stream is FIXED (valid_design_seed), so every seed/policy sees
    the same validation trajectory shape; execution uses the seed's nominal
    controller and its own noise stream. Returns (W_val_base, y_val_meas).
    """
    rng_design = np.random.default_rng(settings.valid_design_seed)
    ref = design_reference("smooth_random", rng_design, traj_cfg, fmodel, beta_ctrl)
    rng_exec = np.random.default_rng([seed, 13])
    execu = execute_reference(fmodel, *ref, beta_ctrl, exec_cfg, rng_exec)
    _, W_base, y = build_observations(execu, proj, fmodel, exec_cfg)
    return W_base, y


def _finalize_metrics(
    policy: str, seed: int, proj, fmodel, post, execu_stats: dict,
    a_true: np.ndarray, W_holdout_base: np.ndarray,
    W_val_base: np.ndarray, y_val: np.ndarray,
) -> dict:
    mu = post.mean()
    resid_hold = W_holdout_base @ (mu - a_true)
    resid_val = W_val_base @ mu - y_val
    return {
        "policy": policy,
        "seed": seed,
        "logdet_cov": post.logdet_cov(),
        "trace_cov": post.trace_cov(),
        "alpha_rmse": alpha_rmse(mu, a_true),
        "holdout_torque_rmse": float(np.sqrt(np.mean(resid_hold**2))),
        "validation_torque_rmse": float(np.sqrt(np.mean(resid_val**2))),
        **execu_stats,
    }


def _exec_stats(logs: list[dict]) -> dict:
    """Aggregate executability evidence over the policy's executed rollouts."""
    q = np.vstack([e["q_true"] for e in logs])
    qd = np.vstack([e["qd_true"] for e in logs])
    tau = np.vstack([e["tau_meas"] for e in logs])
    n_total = sum(e["n_steps"] for e in logs)
    return {
        "fault": bool(any(e["fault"] for e in logs)),
        "tracking_rmse": float(np.sqrt(np.mean([e["tracking_rmse"] ** 2 for e in logs]))),
        "saturation_fraction": float(
            sum(e["saturation_fraction"] * e["n_steps"] for e in logs) / max(1, n_total)
        ),
        "max_abs_q": float(np.max(np.abs(q))),
        "max_abs_qd": float(np.max(np.abs(qd))),
        "peak_abs_tau": float(np.max(np.abs(tau))),
    }


def run_policy_seed(
    policy: str,
    proj: S5ProjectionArtifact,
    settings: S5Layer1Settings,
    traj_cfg: S5TrajConfig,
    exec_cfg: ExecutionConfig,
    seed: int,
    W_holdout_base: np.ndarray,
    beta_ctrl: np.ndarray,
    validation: tuple[np.ndarray, np.ndarray],
) -> dict:
    fmodel = proj.model()
    a_true = alpha_true_ext(proj, fmodel)
    a_prior = proj.V_base @ beta_ctrl
    post = GaussianPosterior.isotropic_prior(a_prior, prior_std=settings.prior_std)

    rng_design = np.random.default_rng([seed, stable_seed(policy)])
    rng_exec = np.random.default_rng([seed, 3])

    obs: list[tuple[np.ndarray, np.ndarray]] = []
    if policy == "segment_active_ig":
        # Interleaved design -> execute -> measure -> update loop. The loop
        # posterior (sensor-noise sigma) drives the segment scoring only; the
        # reported posterior is rebuilt below with the residual-estimated
        # noise, identically for every policy.
        ctx = DesignContext(
            V_base=proj.V_base, sigma_tau=exec_cfg.sigma_tau,
            precision=lambda: post.Lambda.copy(),
        )
        seg_exec_cfg = dataclasses.replace(exec_cfg, edge_trim=5)
        q_ref, qd_ref = np.asarray(traj_cfg.q_start, float), np.zeros(2)
        q_pl, qd_pl = q_ref.copy(), qd_ref.copy()
        logs = []
        for k in range(traj_cfg.n_segments):
            seg = design_next_segment(
                policy, q_ref, qd_ref, k, rng_design, traj_cfg, fmodel, beta_ctrl, ctx=ctx
            )
            execu = execute_reference(
                fmodel, *seg, beta_ctrl, seg_exec_cfg, rng_exec,
                q_init=q_pl, qd_init=qd_pl,
            )
            logs.append(execu)
            q_pl, qd_pl = execu["q_true"][-1], execu["qd_true"][-1]
            q_ref, qd_ref = seg[0][-1], seg[1][-1]
            _, W_base, y = build_observations(execu, proj, fmodel, seg_exec_cfg)
            obs.append((W_base, y))
            post.update(W_base, y, exec_cfg.sigma_tau)
            if execu["fault"]:
                break
        execu_stats = _exec_stats(logs)
    else:
        ctx = None
        if policy == "segment_fim_greedy":
            ctx = DesignContext(V_base=proj.V_base, sigma_tau=exec_cfg.sigma_tau)
        ref = design_reference(policy, rng_design, traj_cfg, fmodel, beta_ctrl, ctx=ctx)
        execu = execute_reference(fmodel, *ref, beta_ctrl, exec_cfg, rng_exec)
        _, W_base, y = build_observations(execu, proj, fmodel, exec_cfg)
        obs.append((W_base, y))
        execu_stats = _exec_stats([execu])

    # Reported posterior: two-pass noise estimation, the standard hardware
    # practice. The measured-state regressor makes the effective observation
    # noise (sensor noise + acceleration-estimation error) larger than the
    # torque-sensor sigma alone; using the raw sensor sigma would make the
    # posterior overconfident. First pass fits with the sensor sigma, the
    # residual RMSE gives sigma_eff, and the reported posterior uses sigma_eff.
    # This needs no ground truth, so a robot can do exactly the same.
    W_all = np.vstack([w for w, _ in obs])
    y_all = np.concatenate([yy for _, yy in obs])
    post = GaussianPosterior.isotropic_prior(a_prior, prior_std=settings.prior_std)
    if y_all.size:
        pass1 = GaussianPosterior.isotropic_prior(a_prior, prior_std=settings.prior_std)
        pass1.update(W_all, y_all, exec_cfg.sigma_tau)
        resid = y_all - W_all @ pass1.mean()
        sigma_eff = float(max(exec_cfg.sigma_tau, np.sqrt(np.mean(resid**2))))
        post.update(W_all, y_all, sigma_eff)
    else:  # all rows velocity-filtered: no usable data, stay at the prior
        sigma_eff = exec_cfg.sigma_tau

    row = _finalize_metrics(
        policy, seed, proj, fmodel, post, execu_stats,
        a_true, W_holdout_base, validation[0], validation[1],
    )
    row["sigma_eff"] = sigma_eff
    return row


METRIC_KEYS = (
    "logdet_cov", "trace_cov", "alpha_rmse", "holdout_torque_rmse",
    "validation_torque_rmse", "sigma_eff", "tracking_rmse", "saturation_fraction",
    "max_abs_q", "max_abs_qd", "peak_abs_tau",
)


def run_closed_loop_ablation(
    proj: S5ProjectionArtifact,
    settings: S5Layer1Settings,
    traj_cfg: S5TrajConfig,
    exec_cfg: ExecutionConfig,
    n_seeds: int,
    policies: tuple[str, ...] = SEGMENT_POLICIES,
) -> dict:
    fmodel = proj.model()
    W_holdout_base = build_holdout(proj, fmodel, settings)
    by_seed: list[dict] = []
    for seed in range(n_seeds):
        rng_nom = np.random.default_rng([seed, 5])
        beta_ctrl = nominal_beta_ext(fmodel, settings, rng_nom)
        validation = execute_validation(
            proj, fmodel, beta_ctrl, traj_cfg, exec_cfg, settings, seed
        )
        for policy in policies:
            by_seed.append(
                run_policy_seed(
                    policy, proj, settings, traj_cfg, exec_cfg, seed,
                    W_holdout_base, beta_ctrl, validation,
                )
            )
    summary = {}
    for policy in policies:
        rows = [r for r in by_seed if r["policy"] == policy]
        summary[policy] = {
            "n_seeds": len(rows),
            "n_faults": int(sum(r["fault"] for r in rows)),
            **{
                k: {
                    "mean": float(np.mean([r[k] for r in rows])),
                    "std": float(np.std([r[k] for r in rows])),
                }
                for k in METRIC_KEYS
            },
        }
    return {"by_seed": by_seed, "summary": summary}
