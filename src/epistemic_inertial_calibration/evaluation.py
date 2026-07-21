"""Shared deterministic trajectories and evaluation metrics for the S5 pipeline."""

from __future__ import annotations

import zlib

import numpy as np


def stable_seed(name: str) -> int:
    """Return a deterministic per-condition seed component."""
    return zlib.crc32(name.encode("utf-8")) % 9973


def alpha_rmse(mu: np.ndarray, alpha_true: np.ndarray) -> float:
    """Return RMSE in identifiable base coordinates."""
    return float(np.sqrt(np.mean((mu - alpha_true) ** 2)))


def minimum_jerk_trajectory(q_start, q_goal, n_steps, dt, reach_frac=0.6):
    """Construct a minimum-jerk reach followed by a hold period."""
    q_start = np.asarray(q_start, float)
    q_goal = np.asarray(q_goal, float)
    n = q_start.size
    qd, qdd = np.zeros((n_steps, n)), np.zeros((n_steps, n))
    q = np.zeros((n_steps, n))
    reach_steps = max(1, int(reach_frac * n_steps))
    duration = reach_steps * dt
    for k in range(n_steps):
        if k < reach_steps:
            u = k / reach_steps
            s = 10 * u**3 - 15 * u**4 + 6 * u**5
            sd = (30 * u**2 - 60 * u**3 + 30 * u**4) / duration
            sdd = (60 * u - 180 * u**2 + 120 * u**3) / duration**2
            q[k] = q_start + s * (q_goal - q_start)
            qd[k] = sd * (q_goal - q_start)
            qdd[k] = sdd * (q_goal - q_start)
        else:
            q[k] = q_goal
    return q, qd, qdd


def auc(scores, labels) -> float:
    """Compute the AUC for a score where larger values predict failure."""
    values = np.asarray(scores, float)
    labels = np.asarray(labels, dtype=bool)
    finite = np.isfinite(values)
    replacement = np.max(values[finite]) + 1.0 if np.any(finite) else 0.0
    values = np.where(finite, values, replacement)
    positive, negative = values[labels], values[~labels]
    if positive.size == 0 or negative.size == 0:
        return float("nan")

    classes = np.concatenate([np.ones(positive.size), np.zeros(negative.size)])
    combined = np.concatenate([positive, negative])
    order = np.argsort(combined)
    sorted_values = combined[order]
    ranks = np.empty(combined.size)
    i = 0
    while i < combined.size:
        j = i
        while j + 1 < combined.size and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    positive_ranks = ranks[classes == 1].sum()
    return float(
        (positive_ranks - positive.size * (positive.size + 1) / 2)
        / (positive.size * negative.size)
    )


def bootstrap_auc_ci(scores, labels, n_boot=2000, seed=0, alpha=0.05):
    """Return a point AUC and percentile bootstrap confidence interval."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, dtype=bool)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_boot):
        indices = rng.integers(0, scores.size, scores.size)
        estimate = auc(scores[indices], labels[indices])
        if np.isfinite(estimate):
            estimates.append(estimate)
    low, high = np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "auc": auc(scores, labels),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_boot": len(estimates),
    }


def bootstrap_auc_diff_ci(scores_a, scores_b, labels, n_boot=2000, seed=0, alpha=0.05):
    """Return a paired-bootstrap confidence interval for an AUC difference."""
    scores_a = np.asarray(scores_a, float)
    scores_b = np.asarray(scores_b, float)
    labels = np.asarray(labels, dtype=bool)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n_boot):
        indices = rng.integers(0, labels.size, labels.size)
        auc_a = auc(scores_a[indices], labels[indices])
        auc_b = auc(scores_b[indices], labels[indices])
        if np.isfinite(auc_a) and np.isfinite(auc_b):
            differences.append(auc_a - auc_b)
    low, high = np.percentile(differences, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "diff": auc(scores_a, labels) - auc(scores_b, labels),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_boot": len(differences),
    }


def stratified_auc(rows, strat_key, score_key, edges, n_boot=2000, seed=0):
    """Compute failure-prediction AUC within intervals of a stratifying score."""
    strat_values = np.asarray([row[strat_key] for row in rows], float)
    result = {
        "strat_key": strat_key,
        "score_key": score_key,
        "edges": list(map(float, edges)),
        "strata": [],
    }
    bounds = [-np.inf, *edges, np.inf]
    for low, high in zip(bounds[:-1], bounds[1:]):
        mask = (strat_values > low) & (strat_values <= high)
        subset = [row for row, selected in zip(rows, mask) if selected]
        labels = [bool(row["failure"]) for row in subset]
        entry = {
            "range": [
                None if not np.isfinite(low) else float(low),
                None if not np.isfinite(high) else float(high),
            ],
            "n": len(subset),
            "n_fail": int(sum(labels)),
            "failure_rate": sum(labels) / len(subset) if subset else None,
            "auc": None,
        }
        if len(subset) >= 20 and 0 < sum(labels) < len(subset):
            scores = [row[score_key] for row in subset]
            entry["auc"] = bootstrap_auc_ci(scores, labels, n_boot=n_boot, seed=seed)
        result["strata"].append(entry)
    return result
