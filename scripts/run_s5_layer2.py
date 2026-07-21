"""S5-3 runner: dynamic Layer-2 execution on the friction 4R plant.

1. Numeric base floor of the friction-extended 4R regressor (rank sweep,
   seeds 0-2); the projection is frozen for the experiment.
2. Closed-loop Layer-2 comparison: task-compatible null-space excitation vs
   the executable free reference, all executed with FF(nominal)+PD on the
   true friction plant; identification from measured data only; task error
   measured on the EXECUTED motion.

Generates under artifacts/s5_layer2_closed_loop/:
  config.json, rank_summary.json, metrics_by_policy.json,
  gap_fractions.json, results_summary.md   (results_by_seed.csv gitignored)

Run from the repository root:
  uv run python scripts/run_s5_layer2.py            # quick (N_SEEDS=5)
  uv run python scripts/run_s5_layer2.py --full     # full (N_SEEDS=50)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import Counter
from pathlib import Path

import numpy as np

from epistemic_inertial_calibration.planar_nr import PlanarNRModel
from epistemic_inertial_calibration.s5_layer2 import (
    S5_LAYER2_POLICIES,
    ExecutionConfigNR,
    S5Layer2Settings,
    default_friction_nr,
    gap_fractions_ext,
    run_layer2_closed_loop,
    stacked_regressor_ext_nr,
)
from epistemic_inertial_calibration.tasks import Task

REPO = Path(__file__).resolve().parent.parent

THRESHOLDS = [10.0**e for e in range(-12, -3)]
SEEDS = (0, 1, 2)
N_RANK_SAMPLES = 2000


def rank_sweep(fm) -> dict:
    n = fm.n
    ranks_by_seed = {}
    sv_ref = None
    for s in SEEDS:
        rng = np.random.default_rng(s)
        states = np.hstack([
            rng.uniform(-np.pi, np.pi, (N_RANK_SAMPLES, n)),
            rng.uniform(-2.0, 2.0, (N_RANK_SAMPLES, n)),
            rng.uniform(-4.0, 4.0, (N_RANK_SAMPLES, n)),
        ])
        W = stacked_regressor_ext_nr(states, fm)
        sv = np.linalg.svd(W, compute_uv=False)
        if sv_ref is None:
            sv_ref = sv
        ranks_by_seed[s] = [int(np.sum(sv > sv[0] * t)) for t in THRESHOLDS]
    flat = [r for rl in ranks_by_seed.values() for r in rl]
    chosen = Counter(flat).most_common(1)[0][0]
    plateau = [t for t, r in zip(THRESHOLDS, ranks_by_seed[SEEDS[0]]) if r == chosen]
    thr = float(plateau[len(plateau) // 2]) if plateau else 1e-8
    return {
        "chosen_rank": chosen, "chosen_threshold": thr,
        "stable": all(r == chosen for r in flat),
        "gap": [float(sv_ref[chosen]) if chosen < sv_ref.size else 0.0,
                float(sv_ref[chosen - 1])],
        "ranks_by_seed": {str(k): v for k, v in ranks_by_seed.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "artifacts" / "s5_layer2_closed_loop")
    ap.add_argument("--full", action="store_true", help="N_SEEDS=50 (default quick: 5)")
    ap.add_argument("--date", type=str, default=dt.date.today().isoformat())
    args = ap.parse_args()
    n_seeds = 50 if args.full else 5

    fm = default_friction_nr(PlanarNRModel())
    sweep = rank_sweep(fm)
    print(f"4R extended base rank = {sweep['chosen_rank']} / {3 * fm.n + 2 * fm.n} "
          f"(stable={sweep['stable']})")

    # Frozen projection at the plateau threshold (seed 0 states).
    rng = np.random.default_rng(SEEDS[0])
    n = fm.n
    states = np.hstack([
        rng.uniform(-np.pi, np.pi, (N_RANK_SAMPLES, n)),
        rng.uniform(-2.0, 2.0, (N_RANK_SAMPLES, n)),
        rng.uniform(-4.0, 4.0, (N_RANK_SAMPLES, n)),
    ])
    W = stacked_regressor_ext_nr(states, fm)
    _, sv, Vt = np.linalg.svd(W, full_matrices=False)
    rank = int(np.sum(sv > sv[0] * sweep["chosen_threshold"]))
    assert rank == sweep["chosen_rank"]
    V_base = Vt[:rank, :]

    # 2 s hold at the 500 Hz control period (dt must match ExecutionConfigNR).
    task = Task(kind="hold", n_steps=1000, dt=2e-3)
    settings = S5Layer2Settings()
    exec_cfg = ExecutionConfigNR()
    result = run_layer2_closed_loop(fm, V_base, task, settings, exec_cfg, n_seeds)
    gaps = gap_fractions_ext(result["summary"])

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "base_projection_ext_4r.npz", V_base=V_base, rank=rank,
             rel_threshold=sweep["chosen_threshold"])
    (args.out / "rank_summary.json").write_text(json.dumps(sweep, indent=2))
    (args.out / "config.json").write_text(json.dumps({
        "mode": "full" if args.full else "quick", "n_seeds": n_seeds,
        "policies": list(S5_LAYER2_POLICIES),
        "task": {"kind": task.kind, "n_steps": task.n_steps, "dt": task.dt},
        "friction": {"fv": list(fm.fv), "fc": list(fm.fc), "eps": fm.eps},
        "execution": dict(exec_cfg.__dict__),
        "settings": dict(settings.__dict__),
        "rank": rank, "generated_date": args.date,
    }, indent=2))
    (args.out / "metrics_by_policy.json").write_text(
        json.dumps(result["summary"], indent=2))
    (args.out / "gap_fractions.json").write_text(json.dumps(gaps, indent=2))
    rows = result["by_seed"]
    with (args.out / "results_by_seed.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    md = [
        "# S5 Layer-2 closed-loop comparison (friction 4R plant)",
        "",
        f"generated: {args.date} | N_SEEDS={n_seeds} | extended base rank {rank}",
        "",
        "| policy | logdet_cov | alpha_rmse | holdout_rmse | ee_err_max | faults |",
        "|---|---|---|---|---|---|",
    ]
    for p, s in result["summary"].items():
        md.append(
            f"| {p} | {s['logdet_cov']['mean']:.2f} | {s['alpha_rmse']['mean']:.3e} | "
            f"{s['holdout_torque_rmse']['mean']:.3e} | {s['ee_error_max']['mean']:.4f} | "
            f"{s['n_faults']} |"
        )
    md += ["", "Gap fractions (share of the free-reference improvement recovered):", ""]
    for metric, d in gaps.items():
        for p, v in d.items():
            md.append(f"- {metric} / {p}: {v:.3f}")
    md.append("")
    (args.out / "results_summary.md").write_text("\n".join(md))

    print(f"S5 Layer-2 ({'full' if args.full else 'quick'}, N={n_seeds}) -> {args.out}")
    for p, s in result["summary"].items():
        print(f"  {p:24s} logdet={s['logdet_cov']['mean']:8.2f}  "
              f"alpha_rmse={s['alpha_rmse']['mean']:.3e}  "
              f"ee_max={s['ee_error_max']['mean']:.4f}  faults={s['n_faults']}")
    for metric, d in gaps.items():
        print(f"  gap[{metric}]: " + ", ".join(f"{p}={v:.2f}" for p, v in d.items()))


if __name__ == "__main__":
    main()
