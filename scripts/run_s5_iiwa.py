"""S5-5 runner: iiwa 7-DoF hardware-executable pipeline.

1. Extended base floor (rigid 70 + armature 7 + friction 14 = 91 params)
   inside the URDF/drake hardware limits; frozen projection.
2. Layer-2: executed task-compatible null-space excitation vs the executable
   free reference (closed loop on the friction+armature plant).
3. Layer-3: velocity-aware terminal feasibility grid with fault labels,
   AUC + go/no-go threshold metrics.

Generates under artifacts/s5_iiwa/:
  base_projection_ext.npz, rank_summary.json
  layer2_metrics.json, layer2_summary.md, layer2_results_by_seed.csv
  layer3_auc.json, layer3_threshold.json, layer3_summary.md,
  layer3_results_by_run.csv

Run from the repository root:
  uv run python scripts/run_s5_iiwa.py            # quick (N=5 L3, N=3 L2)
  uv run python scripts/run_s5_iiwa.py --full     # full (N=50 L3, N=20 L2)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import Counter
from pathlib import Path

import numpy as np

from epistemic_inertial_calibration.evaluation import (
    bootstrap_auc_ci,
    bootstrap_auc_diff_ci,
)
from epistemic_inertial_calibration.s5_iiwa import (
    IIWA_LAYER2_POLICIES,
    load_friction_iiwa,
    run_iiwa_layer2,
    run_iiwa_terminal_grid,
    sample_states_hw,
    stacked_regressor_ext_iiwa,
)

REPO = Path(__file__).resolve().parent.parent
THRESHOLDS = [10.0**e for e in range(-10, -4)]
SEEDS = (0, 1, 2)

L3_SCORES = ("feasibility_risk_ext", "feasibility_risk_ext_est",
             "feasibility_risk_tau_only", "calibration_alpha_rmse")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "artifacts" / "s5_iiwa")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--date", type=str, default=dt.date.today().isoformat())
    args = ap.parse_args()
    n_seeds_l3 = 50 if args.full else 5
    n_seeds_l2 = 20 if args.full else 3
    args.out.mkdir(parents=True, exist_ok=True)

    fm = load_friction_iiwa()

    # --- base floor -------------------------------------------------------
    ranks_by_seed = {}
    sv_ref = None
    for s in SEEDS:
        W = stacked_regressor_ext_iiwa(sample_states_hw(fm, 1000, seed=s), fm)
        sv = np.linalg.svd(W, compute_uv=False)
        if sv_ref is None:
            sv_ref = sv
        ranks_by_seed[s] = [int(np.sum(sv > sv[0] * t)) for t in THRESHOLDS]
    flat = [r for rl in ranks_by_seed.values() for r in rl]
    rank = Counter(flat).most_common(1)[0][0]
    stable = all(r == rank for r in flat)
    W = stacked_regressor_ext_iiwa(sample_states_hw(fm, 1000, seed=SEEDS[0]), fm)
    _, sv, Vt = np.linalg.svd(W, full_matrices=False)
    V_base = Vt[:rank, :]
    np.savez(args.out / "base_projection_ext.npz", V_base=V_base, rank=rank)
    (args.out / "rank_summary.json").write_text(json.dumps({
        "rank": rank, "n_params": fm.n_params, "stable": stable,
        "thresholds": THRESHOLDS,
        "ranks_by_seed": {str(k): v for k, v in ranks_by_seed.items()},
        "gap": [float(sv[rank]) if rank < sv.size else 0.0, float(sv[rank - 1])],
        "note": "within URDF position/velocity + drake acceleration limits; "
                "91 = 70 rigid + 7 armature + 14 friction",
        "generated_date": args.date,
    }, indent=2))
    print(f"iiwa extended base rank = {rank} / {fm.n_params} (stable={stable})")

    # --- Layer 2 ----------------------------------------------------------
    l2 = run_iiwa_layer2(fm, V_base, n_seeds=n_seeds_l2)
    with (args.out / "layer2_results_by_seed.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=sorted(l2["by_seed"][0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(l2["by_seed"])
    (args.out / "layer2_metrics.json").write_text(json.dumps(l2["summary"], indent=2))
    md = [
        "# S5 iiwa Layer-2 (executed task-compatible excitation)",
        "",
        f"generated: {args.date} | N_SEEDS={n_seeds_l2} | rank {rank}",
        "",
        "| policy | logdet_cov | alpha_rmse | holdout_rmse | ee_max | faults |",
        "|---|---|---|---|---|---|",
    ]
    for p, s in l2["summary"].items():
        md.append(
            f"| {p} | {s['logdet_cov']['mean']:.2f} | {s['alpha_rmse']['mean']:.3e} | "
            f"{s['holdout_torque_rmse']['mean']:.3e} | {s['ee_error_max']['mean']:.4f} | "
            f"{s['n_faults']} |"
        )
    (args.out / "layer2_summary.md").write_text("\n".join(md) + "\n")
    for p, s in l2["summary"].items():
        print(f"  L2 {p:20s} logdet={s['logdet_cov']['mean']:8.2f} "
              f"alpha={s['alpha_rmse']['mean']:.3e} faults={s['n_faults']}")

    # --- Layer 3 ----------------------------------------------------------
    rows = run_iiwa_terminal_grid(fm, V_base, n_seeds=n_seeds_l3)
    with (args.out / "layer3_results_by_run.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    fail = [r["failure"] for r in rows]
    aucs = {k: bootstrap_auc_ci([r[k] for r in rows], fail, n_boot=args.n_boot, seed=0)
            for k in L3_SCORES}
    diffs = {
        "ext_minus_calibration": bootstrap_auc_diff_ci(
            [r["feasibility_risk_ext"] for r in rows],
            [r["calibration_alpha_rmse"] for r in rows], fail,
            n_boot=args.n_boot, seed=1),
        "ext_minus_tau_only": bootstrap_auc_diff_ci(
            [r["feasibility_risk_ext"] for r in rows],
            [r["feasibility_risk_tau_only"] for r in rows], fail,
            n_boot=args.n_boot, seed=2),
    }

    def thr_metrics(key, thr=1.0):
        s = np.array([r[key] for r in rows], float)
        y = np.array(fail, bool)
        return {
            "missed_failures": int(np.sum((s <= thr) & y)),
            "miss_rate": float(np.sum((s <= thr) & y) / max(1, np.sum(y))),
            "false_alarms": int(np.sum((s > thr) & ~y)),
        }

    thr = {k: thr_metrics(k) for k in
           ("feasibility_risk_ext", "feasibility_risk_ext_est",
            "feasibility_risk_tau_only")}
    n_fault = int(np.sum([r["fault"] for r in rows]))
    (args.out / "layer3_auc.json").write_text(json.dumps(
        {"auc": aucs, "auc_diff": diffs, "n_runs": len(rows),
         "n_failures": int(np.sum(fail)), "n_velocity_faults": n_fault}, indent=2))
    (args.out / "layer3_threshold.json").write_text(json.dumps(thr, indent=2))
    md = ["# S5 iiwa Layer-3 (velocity-aware terminal feasibility)", "",
          f"generated: {args.date} | N_SEEDS={n_seeds_l3} | runs={len(rows)} | "
          f"failures={int(np.sum(fail))} | velocity faults={n_fault}", "",
          "| score | AUC | 95% CI |", "|---|---|---|"]
    for k in L3_SCORES:
        a = aucs[k]
        md.append(f"| {k} | {a['auc']:.3f} | [{a['ci_low']:.3f}, {a['ci_high']:.3f}] |")
    md += ["", "## Go/no-go at rho = 1", "",
           "| score | missed failures | miss rate | false alarms |", "|---|---|---|---|"]
    for k, t in thr.items():
        md.append(f"| {k} | {t['missed_failures']} | {t['miss_rate']:.3f} | {t['false_alarms']} |")
    (args.out / "layer3_summary.md").write_text("\n".join(md) + "\n")
    for k in L3_SCORES:
        a = aucs[k]
        print(f"  L3 {k:28s} AUC={a['auc']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}]")
    for k, t in thr.items():
        print(f"  L3 go/no-go {k:26s} miss={t['miss_rate']:.3f} false_alarms={t['false_alarms']}")

    (args.out / "s5_config.json").write_text(json.dumps({
        "mode": "full" if args.full else "quick",
        "n_seeds_layer3": n_seeds_l3, "n_seeds_layer2": n_seeds_l2,
        "policies_layer2": list(IIWA_LAYER2_POLICIES),
        "friction": {"fv": list(fm.fv), "fc": list(fm.fc), "eps": fm.eps},
        "armature": list(fm.armature),
        "generated_date": args.date,
    }, indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
