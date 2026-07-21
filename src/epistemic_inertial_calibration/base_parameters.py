"""Base inertial parameter extraction and rank/conditioning analysis (S0).

From the SVD of the stacked regressor W (2N x 6), compute:
- the numerical rank (= number of base parameters),
- the projection onto the base subspace (right singular vectors),
- a rank threshold sweep (mandatory, to avoid threshold artifacts).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .planar2r import BETA_LABELS, Planar2RModel, stacked_regressor


@dataclass
class SamplingConfig:
    """Sampling configuration for the excitation regime.

    regime: "static" | "quasi_static" | "dynamic"
        static       : qd = qdd = 0 (pose only)
        quasi_static : qd, qdd perturbed by small noise
        dynamic      : qd, qdd excited sufficiently
    """

    regime: str = "dynamic"
    n_samples: int = 1000
    seed: int = 0
    q_range: float = np.pi  # uniform range of joint angles [-q_range, q_range]
    vel_scale: float = 2.0  # velocity scale for "dynamic"
    acc_scale: float = 4.0  # acceleration scale for "dynamic"
    quasi_scale: float = 1e-3  # small-excitation scale for "quasi_static"


def sample_states(cfg: SamplingConfig) -> np.ndarray:
    """Generate (n_samples x 6) state samples [q1,q2,dq1,dq2,ddq1,ddq2]."""
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_samples
    q = rng.uniform(-cfg.q_range, cfg.q_range, size=(n, 2))

    if cfg.regime == "static":
        qd = np.zeros((n, 2))
        qdd = np.zeros((n, 2))
    elif cfg.regime == "quasi_static":
        qd = rng.normal(0.0, cfg.quasi_scale, size=(n, 2))
        qdd = rng.normal(0.0, cfg.quasi_scale, size=(n, 2))
    elif cfg.regime == "dynamic":
        qd = rng.uniform(-cfg.vel_scale, cfg.vel_scale, size=(n, 2))
        qdd = rng.uniform(-cfg.acc_scale, cfg.acc_scale, size=(n, 2))
    else:
        raise ValueError(f"unknown regime: {cfg.regime!r}")

    return np.hstack([q, qd, qdd])


@dataclass
class RankResult:
    singular_values: np.ndarray
    rank: int
    threshold: float  # relative threshold (ratio to sigma_max)
    abs_threshold: float  # absolute threshold sigma_max * threshold
    cond_base: float  # condition number of the base subspace sigma_max / sigma_min(accepted)
    min_accepted_sv: float
    max_rejected_sv: float
    margin: float  # min_accepted / max_rejected (larger = clearer rank)


def analyze_rank(W: np.ndarray, rel_threshold: float = 1e-8) -> RankResult:
    """Compute the numerical rank and conditioning from the SVD of W."""
    sv = np.linalg.svd(W, compute_uv=False)
    sv = np.asarray(sv, dtype=float)
    smax = float(sv[0])
    abs_thr = smax * rel_threshold
    accepted = sv[sv > abs_thr]
    rejected = sv[sv <= abs_thr]
    rank = int(accepted.size)
    min_acc = float(accepted[-1]) if accepted.size else 0.0
    max_rej = float(rejected[0]) if rejected.size else 0.0
    cond_base = float(smax / min_acc) if min_acc > 0 else np.inf
    margin = float(min_acc / max_rej) if max_rej > 0 else np.inf
    return RankResult(
        singular_values=sv,
        rank=rank,
        threshold=rel_threshold,
        abs_threshold=abs_thr,
        cond_base=cond_base,
        min_accepted_sv=min_acc,
        max_rejected_sv=max_rej,
        margin=margin,
    )


@dataclass
class ThresholdSweep:
    thresholds: list[float]
    ranks: list[int]
    singular_values: np.ndarray
    min_accepted: list[float]
    max_rejected: list[float]
    margins: list[float]
    cond_base: list[float]
    plateau_rank: int | None = None
    plateau_thresholds: list[float] = field(default_factory=list)


def threshold_sweep(W: np.ndarray, thresholds: list[float] | None = None) -> ThresholdSweep:
    """Sweep the rank threshold and record the rank and conditioning at each value."""
    if thresholds is None:
        thresholds = [10.0**e for e in range(-12, -1)]  # 1e-12 .. 1e-2
    sv = np.linalg.svd(W, compute_uv=False)
    ranks, min_acc, max_rej, margins, conds = [], [], [], [], []
    for thr in thresholds:
        r = analyze_rank(W, rel_threshold=thr)
        ranks.append(r.rank)
        min_acc.append(r.min_accepted_sv)
        max_rej.append(r.max_rejected_sv)
        margins.append(r.margin)
        conds.append(r.cond_base)

    # Take the widest plateau (longest run of identical rank) as the chosen candidate.
    plateau_rank, plateau_thrs = _widest_plateau(thresholds, ranks)
    return ThresholdSweep(
        thresholds=list(thresholds),
        ranks=ranks,
        singular_values=np.asarray(sv, dtype=float),
        min_accepted=min_acc,
        max_rejected=max_rej,
        margins=margins,
        cond_base=conds,
        plateau_rank=plateau_rank,
        plateau_thresholds=plateau_thrs,
    )


def _widest_plateau(thresholds: list[float], ranks: list[int]) -> tuple[int | None, list[float]]:
    best_rank: int | None = None
    best: list[float] = []
    i = 0
    n = len(ranks)
    while i < n:
        j = i
        while j + 1 < n and ranks[j + 1] == ranks[i]:
            j += 1
        run = thresholds[i : j + 1]
        if len(run) > len(best):
            best = run
            best_rank = ranks[i]
        i = j + 1
    return best_rank, best


@dataclass
class BaseProjection:
    """Projection onto the base subspace and its provenance metadata."""

    rank: int
    V_base: np.ndarray  # (rank x 6) right singular vectors (projection to base coords)
    singular_values: np.ndarray
    beta_labels: tuple[str, ...]
    # Provenance metadata.
    regime: str
    n_samples: int
    seed: int
    rel_threshold: float
    gravity_in_plane: bool
    l1: float
    g: float


def extract_base_projection(
    model: Planar2RModel,
    cfg: SamplingConfig,
    rel_threshold: float = 1e-8,
) -> tuple[BaseProjection, RankResult, np.ndarray]:
    """Extract the base projection to be frozen and reused in S1.

    Returns: (projection, rank_result, W)
    """
    samples = sample_states(cfg)
    W = stacked_regressor(samples, model)
    _, sv, Vt = np.linalg.svd(W, full_matrices=False)
    rr = analyze_rank(W, rel_threshold=rel_threshold)
    V_base = Vt[: rr.rank, :].copy()
    proj = BaseProjection(
        rank=rr.rank,
        V_base=V_base,
        singular_values=np.asarray(sv, dtype=float),
        beta_labels=BETA_LABELS,
        regime=cfg.regime,
        n_samples=cfg.n_samples,
        seed=cfg.seed,
        rel_threshold=rel_threshold,
        gravity_in_plane=model.gravity_in_plane,
        l1=model.l1,
        g=model.g,
    )
    return proj, rr, W


def expected_base_dim(model: Planar2RModel) -> int:
    """Ground-truth base dimension (delegates to symbolic exact rank; threshold-free).

    For S0 this is vertical=4 / horizontal=3 (settled in the 2026-06-18 blocker). The
    value is not hardcoded; it is derived from the symbolic rank and cross-checked
    against the numerical rank. A remaining mismatch is itself an S0 finding.
    """
    from .planar2r import analytic_base_dim_symbolic

    return analytic_base_dim_symbolic(model)
