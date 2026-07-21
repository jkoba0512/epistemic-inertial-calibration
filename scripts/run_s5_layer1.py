"""Runner for the S5 closed-loop Layer-1 comparison (executable pipeline).

Every policy designs one continuous executable reference (envelope + nominal
torque budget respected at design time), executes it closed-loop on the true
friction plant, and identifies the extended base parameters from measured
data only. Generates under artifacts/s5_layer1_closed_loop/:
  config.json
  metrics_by_policy.json   # per-policy mean/std summary (tracked)
  results_by_seed.csv      # per-seed rows (gitignored: raw)
  results_summary.md       # ranking + executability table (tracked)

Run from the repository root:
  uv run python scripts/run_s5_layer1.py            # quick (N_SEEDS=5)
  uv run python scripts/run_s5_layer1.py --full     # full (N_SEEDS=50)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

from epistemic_inertial_calibration.s5_execution import ExecutionConfig
from epistemic_inertial_calibration.s5_layer1 import (
    METRIC_KEYS,
    S5Layer1Settings,
    run_closed_loop_ablation,
)
from epistemic_inertial_calibration.s5_model import load_s5_projection
from epistemic_inertial_calibration.s5_trajectories import SEGMENT_POLICIES, S5TrajConfig

REPO = Path(__file__).resolve().parent.parent
S5_DIR = REPO / "artifacts" / "s5_base_floor"


def _write_summary_md(path: Path, all_summaries: dict, date: str,
                      traj: S5TrajConfig, execc: ExecutionConfig) -> None:
    md = [
        "# S5 closed-loop Layer-1 comparison - summary",
        "",
        f"generated: {date}",
        f"reference: {traj.n_segments} x {traj.seg_duration} s segments at dt={traj.dt}",
        f"envelope: |q|<={traj.q_range:.3f}, |qd|<={traj.vel_limit}, "
        f"|qdd|<={traj.acc_limit}, nominal |tau|<={traj.torque_margin}*{traj.tau_limit}",
        f"sensors: sigma_q={execc.sigma_q}, sigma_qd={execc.sigma_qd}, "
        f"sigma_tau={execc.sigma_tau}; qdd estimated from measured qd",
        "",
        "All policies are continuous executable trajectories tracked closed-loop",
        "on the true friction plant; identification uses measured data only.",
        "",
    ]
    for name, summary in all_summaries.items():
        ranked = sorted(summary, key=lambda p: summary[p]["alpha_rmse"]["mean"])
        md += [
            f"## {name}",
            "",
            "| rank | policy | alpha_rmse | logdet_cov | holdout_rmse | validation_rmse | faults |",
            "|---|---|---|---|---|---|---|",
        ]
        for i, p in enumerate(ranked, 1):
            s = summary[p]
            md.append(
                f"| {i} | {p} | {s['alpha_rmse']['mean']:.3e} | "
                f"{s['logdet_cov']['mean']:.2f} | {s['holdout_torque_rmse']['mean']:.3e} | "
                f"{s['validation_torque_rmse']['mean']:.3e} | {s['n_faults']} |"
            )
        md += [
            "",
            "Executability (means over seeds):",
            "",
            "| policy | tracking_rmse | q_max | qd_max | tau_peak | saturation |",
            "|---|---|---|---|---|---|",
        ]
        for p in summary:
            s = summary[p]
            md.append(
                f"| {p} | {s['tracking_rmse']['mean']:.3f} | {s['max_abs_q']['mean']:.2f} | "
                f"{s['max_abs_qd']['mean']:.2f} | {s['peak_abs_tau']['mean']:.1f} | "
                f"{s['saturation_fraction']['mean']:.3f} |"
            )
        md.append("")
    path.write_text("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "artifacts" / "s5_layer1_closed_loop")
    ap.add_argument("--full", action="store_true", help="N_SEEDS=50 (default quick: 5)")
    ap.add_argument("--configs", nargs="+", default=["vertical", "horizontal"])
    ap.add_argument("--date", type=str, default=dt.date.today().isoformat())
    args = ap.parse_args()

    n_seeds = 50 if args.full else 5
    settings = S5Layer1Settings()
    traj_cfg = S5TrajConfig()
    exec_cfg = ExecutionConfig()
    args.out.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    all_rows = []
    for name in args.configs:
        proj = load_s5_projection(S5_DIR / f"base_projection_ext_{name}.npz")
        result = run_closed_loop_ablation(
            proj, settings, traj_cfg, exec_cfg, n_seeds=n_seeds
        )
        all_summaries[name] = result["summary"]
        for r in result["by_seed"]:
            all_rows.append({"config": name, **r})
        print(f"[{name}] done ({n_seeds} seeds)")

    (args.out / "config.json").write_text(json.dumps({
        "mode": "full" if args.full else "quick",
        "n_seeds": n_seeds,
        "policies": list(SEGMENT_POLICIES),
        "traj": {k: (list(v) if isinstance(v, tuple) else v)
                 for k, v in traj_cfg.__dict__.items()},
        "execution": dict(exec_cfg.__dict__),
        "settings": dict(settings.__dict__),
        "metric_keys": list(METRIC_KEYS),
        "generated_date": args.date,
    }, indent=2))
    (args.out / "metrics_by_policy.json").write_text(
        json.dumps(all_summaries, indent=2, ensure_ascii=False)
    )
    if all_rows:
        with (args.out / "results_by_seed.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()), lineterminator="\n")
            w.writeheader()
            w.writerows(all_rows)
    _write_summary_md(args.out / "results_summary.md", all_summaries, args.date,
                      traj_cfg, exec_cfg)

    print(f"S5 closed-loop Layer-1 ({'full' if args.full else 'quick'}, N={n_seeds}) -> {args.out}")
    for name, summary in all_summaries.items():
        print(f"\n[{name}] ranking by alpha_rmse:")
        ranked = sorted(summary, key=lambda p: summary[p]["alpha_rmse"]["mean"])
        for i, p in enumerate(ranked, 1):
            s = summary[p]
            print(
                f"  {i}. {p:20s} alpha_rmse={s['alpha_rmse']['mean']:.3e}  "
                f"logdet={s['logdet_cov']['mean']:8.2f}  "
                f"valid_rmse={s['validation_torque_rmse']['mean']:.3e}  "
                f"track={s['tracking_rmse']['mean']:.3f}  faults={s['n_faults']}"
            )


if __name__ == "__main__":
    main()
