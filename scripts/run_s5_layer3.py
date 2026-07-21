"""S5-4 runner: terminal feasibility with velocity limits and fault labels (2R).

Runs the extended Layer-3 grid on the friction plant, then the AUC analysis:
  feasibility_risk_ext_est  primary online score from the estimated parameters
  feasibility_risk_ext      oracle score from the true simulation parameters
  feasibility_risk_tau_only torque-only ablation of the online score (blind to
                            velocity faults; estimated parameters)
  calibration_alpha_rmse    extended base-coordinate calibration error
  holdout_torque_rmse       torque-space calibration error

Generates under artifacts/s5_layer3_terminal/:
  config.json, auc_bootstrap.json, auc_summary.md, stratified_metrics.json,
  nullspace_control.json, results_by_run.csv

Run from the repository root:
  uv run python scripts/run_s5_layer3.py            # N_SEEDS=50
  uv run python scripts/run_s5_layer3.py --n-seeds 5
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np

from epistemic_inertial_calibration.evaluation import (
    bootstrap_auc_ci,
    bootstrap_auc_diff_ci,
    stratified_auc,
)
from epistemic_inertial_calibration.s5_layer3 import run_grid_ext
from epistemic_inertial_calibration.s5_model import load_s5_projection

REPO = Path(__file__).resolve().parent.parent
S5_PROJ = REPO / "artifacts" / "s5_base_floor" / "base_projection_ext_vertical.npz"

RISK_SCORES = (
    "feasibility_risk_ext",
    "feasibility_risk_ext_est",
    "feasibility_risk_tau_only",
    "calibration_alpha_rmse",
    "holdout_torque_rmse",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "artifacts" / "s5_layer3_terminal")
    ap.add_argument("--n-seeds", type=int, default=50)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--date", type=str, default=dt.date.today().isoformat())
    args = ap.parse_args()

    proj = load_s5_projection(S5_PROJ)
    grid = run_grid_ext(proj, n_seeds=args.n_seeds, include_pin_shift=True)
    all_rows = grid["rows"]
    # The balanced factorial sweep is the population for all primary statistics.
    # Null-space pin shifts are a separate paired invariance control; mixing them
    # into the AUC would overweight the cal0/normal/near conditions.
    rows = [r for r in all_rows if not r["pin_shift"]]
    control_rows = [r for r in all_rows if r["pin_shift"]]
    fail = [r["failure"] for r in rows]
    n_fail = int(np.sum(fail))
    n_fault = int(np.sum([r["fault"] for r in rows]))

    # Reproduction check (seed determinism): rerun a small slice and compare.
    check = run_grid_ext(proj, n_seeds=1, include_pin_shift=False)["rows"]
    ref = [r for r in rows if r["seed"] == 0 and not r["pin_shift"]]
    delta = max(
        abs(a["feasibility_risk_ext"] - b["feasibility_risk_ext"])
        + abs(a["hold_error"] - b["hold_error"])
        if np.isfinite(a["hold_error"]) and np.isfinite(b["hold_error"])
        else abs(a["feasibility_risk_ext"] - b["feasibility_risk_ext"])
        for a, b in zip(ref, check)
    )
    if delta > 0:
        raise RuntimeError(f"non-reproducible grid (delta={delta})")

    # Pair every null-space perturbation with its unperturbed cal0 run under the
    # same torque limit, horizon, target, and seed.
    baseline = {
        (r["torque"], r["horizon"], r["target"], r["seed"]): r
        for r in rows if r["cal"] == "cal0"
    }
    pairs = [
        (baseline[(r["torque"], r["horizon"], r["target"], r["seed"])], r)
        for r in control_rows
    ]

    def _max_pair_delta(key: str) -> float:
        vals = [abs(float(a[key]) - float(b[key])) for a, b in pairs]
        return float(max(vals, default=0.0))

    null_control = {
        "n_pairs": len(pairs),
        "outcome_mismatches": int(sum(a["failure"] != b["failure"] for a, b in pairs)),
        "fault_mismatches": int(sum(a["fault"] != b["fault"] for a, b in pairs)),
        "max_abs_delta_base_rmse": _max_pair_delta("calibration_alpha_rmse"),
        "max_abs_delta_holdout_torque_rmse": _max_pair_delta("holdout_torque_rmse"),
        "max_abs_delta_rho_online": _max_pair_delta("feasibility_risk_ext_est"),
        "max_abs_delta_rho_tau": _max_pair_delta("feasibility_risk_tau_only"),
        "max_abs_delta_hold_error": _max_pair_delta("hold_error"),
        "max_abs_delta_hold_velocity": _max_pair_delta("hold_velocity"),
        "max_abs_delta_saturation_fraction": _max_pair_delta("saturation_fraction"),
    }

    aucs = {k: bootstrap_auc_ci([r[k] for r in rows], fail, n_boot=args.n_boot, seed=0)
            for k in RISK_SCORES}
    diffs = {
        "est_minus_calibration": bootstrap_auc_diff_ci(
            [r["feasibility_risk_ext_est"] for r in rows],
            [r["calibration_alpha_rmse"] for r in rows], fail,
            n_boot=args.n_boot, seed=1),
        "est_minus_tau_only": bootstrap_auc_diff_ci(
            [r["feasibility_risk_ext_est"] for r in rows],
            [r["feasibility_risk_tau_only"] for r in rows], fail,
            n_boot=args.n_boot, seed=2),
        "oracle_minus_est": bootstrap_auc_diff_ci(
            [r["feasibility_risk_ext"] for r in rows],
            [r["feasibility_risk_ext_est"] for r in rows], fail,
            n_boot=args.n_boot, seed=3),
    }

    # Sequential reading: within the online-score feasible stratum (rho < 1),
    # does calibration error become predictive?
    strat = stratified_auc(rows, strat_key="feasibility_risk_ext_est",
                           score_key="calibration_alpha_rmse",
                           edges=(1.0, 3.0), n_boot=args.n_boot, seed=4)

    # Decision metrics at the natural go/no-go threshold rho = 1. AUC is
    # rank-based and cannot expose the torque-only score's calibration
    # blindness: velocity-only-infeasible runs keep rho_tau < 1 (score says
    # "go") yet fault and fail on hardware-consistent labels.
    def _threshold_metrics(key: str, thr: float = 1.0) -> dict:
        s = np.array([r[key] for r in rows], float)
        y = np.array(fail, bool)
        missed = int(np.sum((s <= thr) & y))       # score says go, run fails
        false_alarm = int(np.sum((s > thr) & ~y))  # score says no-go, run succeeds
        return {
            "threshold": thr,
            "missed_failures": missed,
            "miss_rate": missed / max(1, int(np.sum(y))),
            "false_alarms": false_alarm,
            "false_alarm_rate": false_alarm / max(1, int(np.sum(~y))),
        }

    thr_metrics = {k: _threshold_metrics(k) for k in
                   ("feasibility_risk_ext", "feasibility_risk_ext_est",
                    "feasibility_risk_tau_only")}

    # Calibration-vs-outcome quadrants (well-calibrated = base alpha RMSE < 0.05).
    cal = np.array([r["calibration_alpha_rmse"] for r in rows])
    yf = np.array(fail, bool)
    well = cal < 0.05
    quadrants = {
        "well_success": int((well & ~yf).sum()),
        "well_fail": int((well & yf).sum()),
        "poor_success": int((~well & ~yf).sum()),
        "poor_fail": int((~well & yf).sum()),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "results_by_run.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(all_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    (args.out / "config.json").write_text(json.dumps({
        "n_seeds": args.n_seeds, "n_runs": len(rows),
        "n_nullspace_control_runs": len(control_rows), "n_boot": args.n_boot,
        "n_failures": n_fail, "n_velocity_faults": n_fault,
        "scores": list(RISK_SCORES),
        "projection": str(S5_PROJ.name), "rank": grid["rank"],
        "generated_date": args.date,
        "note": "S5 Layer-3: friction plant, velocity-limit faults count as "
                "failure; rho = max(rho_tau, rho_vel) on the desired trajectory.",
    }, indent=2))
    (args.out / "auc_bootstrap.json").write_text(json.dumps(
        {"auc": aucs, "auc_diff": diffs}, indent=2))
    (args.out / "threshold_metrics.json").write_text(json.dumps(thr_metrics, indent=2))
    (args.out / "stratified_metrics.json").write_text(json.dumps(strat, indent=2))
    (args.out / "nullspace_control.json").write_text(json.dumps(null_control, indent=2))
    (args.out / "quadrant_summary.md").write_text(
        "# quadrant counts (well-calibrated = base alpha RMSE < 0.05)\n\n"
        "| quadrant | count |\n|---|---|\n"
        + "".join(f"| {k} | {v} |\n" for k, v in quadrants.items()))

    md = ["# S5 Layer-3 AUC summary (higher score predicts failure)", "",
          f"generated: {args.date} | N_SEEDS={args.n_seeds} | runs={len(rows)} | "
          f"failures={n_fail} | velocity faults={n_fault}", "",
          "| score | AUC | 95% CI |", "|---|---|---|"]
    for k in RISK_SCORES:
        a = aucs[k]
        md.append(f"| {k} | {a['auc']:.3f} | [{a['ci_low']:.3f}, {a['ci_high']:.3f}] |")
    md += ["", "## AUC differences (95% CI)", "",
           "| comparison | diff | 95% CI |", "|---|---|---|"]
    for k, d in diffs.items():
        md.append(f"| {k} | {d['diff']:.3f} | [{d['ci_low']:.3f}, {d['ci_high']:.3f}] |")
    md += ["", "## Go/no-go decision at rho = 1", "",
           "| score | missed failures | miss rate | false alarms |",
           "|---|---|---|---|"]
    for k, t in thr_metrics.items():
        md.append(f"| {k} | {t['missed_failures']} | {t['miss_rate']:.3f} | "
                  f"{t['false_alarms']} |")
    md += ["", "## Stratified calibration AUC (by online rho)", ""]
    for s in strat["strata"]:
        a = s["auc"]
        auc_txt = f"{a['auc']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}]" if a else "n/a"
        md.append(f"- rho in {s['range']}: n={s['n']}, failure_rate={s['failure_rate']:.2f}, "
                  f"calibration AUC = {auc_txt}")
    md.append("")
    md += ["", "## Separate null-space invariance control", "",
           f"- paired runs: {null_control['n_pairs']}",
           f"- outcome mismatches: {null_control['outcome_mismatches']}",
           f"- maximum |delta rho_online|: {null_control['max_abs_delta_rho_online']:.3e}",
           f"- maximum holdout-torque RMSE change: "
           f"{null_control['max_abs_delta_holdout_torque_rmse']:.3e}"]
    (args.out / "auc_summary.md").write_text("\n".join(md))

    print(f"S5 Layer-3 (N={args.n_seeds}, runs={len(rows)}, failures={n_fail}, "
          f"velocity faults={n_fault}) -> {args.out}")
    for k in RISK_SCORES:
        a = aucs[k]
        print(f"  {k:28s} AUC={a['auc']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}]")
    for k, t in thr_metrics.items():
        print(f"  go/no-go {k:28s} miss_rate={t['miss_rate']:.3f} "
              f"({t['missed_failures']} missed), false_alarms={t['false_alarms']}")


if __name__ == "__main__":
    main()
