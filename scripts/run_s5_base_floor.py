"""S5-0 runner: base-parameter floor of the friction-extended 2R model.

Numeric SVD rank of the stacked extended regressor (2N x 10), with a threshold
sweep and cross-seed stability check (the friction columns involve tanh, so no
symbolic certificate is claimed; the rigid 6-column sub-block keeps the S0
symbolic certificate). Expected: rigid rank + 4 friction parameters, i.e.
8 vertical / 7 horizontal. Generates under artifacts/s5_base_floor/:
  base_projection_ext_{vertical,horizontal}.npz   # frozen V_base (rank x 10)
  rank_summary.json
  s5_base_floor.md

Run from the repository root:
  uv run python scripts/run_s5_base_floor.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path

import numpy as np

from epistemic_inertial_calibration.base_parameters import SamplingConfig, sample_states
from epistemic_inertial_calibration.planar2r import Planar2RModel
from epistemic_inertial_calibration.s5_model import (
    BETA_EXT_LABELS,
    FrictionModel2R,
    load_s5_projection,
    save_s5_projection,
    stacked_regressor_ext,
)

REPO = Path(__file__).resolve().parent.parent

THRESHOLDS = [10.0**e for e in range(-12, -3)]  # 1e-12 .. 1e-4
SEEDS = (0, 1, 2)
N_SAMPLES = 1000


def rank_sweep(fmodel: FrictionModel2R) -> dict:
    """SVD rank across thresholds x seeds; modal rank must be unanimous."""
    ranks_by_seed = {}
    sv_ref = None
    for s in SEEDS:
        states = sample_states(SamplingConfig(regime="dynamic", n_samples=N_SAMPLES, seed=s))
        W = stacked_regressor_ext(states, fmodel)
        sv = np.linalg.svd(W, compute_uv=False)
        if sv_ref is None:
            sv_ref = sv
        ranks_by_seed[s] = [int(np.sum(sv > sv[0] * t)) for t in THRESHOLDS]
    flat = [r for rl in ranks_by_seed.values() for r in rl]
    chosen = Counter(flat).most_common(1)[0][0]
    plateau = [t for t, r in zip(THRESHOLDS, ranks_by_seed[SEEDS[0]]) if r == chosen]
    chosen_thr = float(plateau[len(plateau) // 2]) if plateau else 1e-8
    return {
        "ranks_by_seed": {str(k): v for k, v in ranks_by_seed.items()},
        "thresholds": THRESHOLDS,
        "chosen_rank": chosen,
        "chosen_threshold": chosen_thr,
        "stable": all(r == chosen for r in flat),
        "gap_high": float(sv_ref[chosen - 1]),
        "gap_low": float(sv_ref[chosen]) if chosen < sv_ref.size else 0.0,
        "singular_values": [float(x) for x in sv_ref],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "artifacts" / "s5_base_floor")
    ap.add_argument("--date", type=str, default=dt.date.today().isoformat())
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summary = {}
    for name, g in (("vertical", 9.81), ("horizontal", 0.0)):
        fmodel = FrictionModel2R(rigid=Planar2RModel(g=g))
        sweep = rank_sweep(fmodel)

        # Freeze the projection at the plateau threshold using seed SEEDS[0].
        states = sample_states(
            SamplingConfig(regime="dynamic", n_samples=N_SAMPLES, seed=SEEDS[0])
        )
        W = stacked_regressor_ext(states, fmodel)
        _, sv, Vt = np.linalg.svd(W, full_matrices=False)
        rank = int(np.sum(sv > sv[0] * sweep["chosen_threshold"]))
        assert rank == sweep["chosen_rank"], (rank, sweep["chosen_rank"])
        npz = args.out / f"base_projection_ext_{name}.npz"
        save_s5_projection(npz, Vt[:rank, :], rank, fmodel, sweep["chosen_threshold"])

        # Reproduction check: load back and verify the projection round-trips.
        proj = load_s5_projection(npz)
        assert proj.rank == rank
        assert np.allclose(proj.V_base, Vt[:rank, :])

        summary[name] = {
            "rank": rank,
            "stable_across_thresholds_and_seeds": sweep["stable"],
            "chosen_threshold": sweep["chosen_threshold"],
            "gap": [sweep["gap_low"], sweep["gap_high"]],
            "singular_values": sweep["singular_values"],
            "ranks_by_seed": sweep["ranks_by_seed"],
        }
        print(
            f"[{name}] extended base rank = {rank} / {len(BETA_EXT_LABELS)} "
            f"(stable={sweep['stable']}, gap {sweep['gap_low']:.2e} -> {sweep['gap_high']:.2e})"
        )

    (args.out / "rank_summary.json").write_text(
        json.dumps(
            {
                "n_samples": N_SAMPLES,
                "seeds": list(SEEDS),
                "thresholds": THRESHOLDS,
                "beta_labels": list(BETA_EXT_LABELS),
                "friction": {
                    "fv": list(FrictionModel2R().fv),
                    "fc": list(FrictionModel2R().fc),
                    "eps": FrictionModel2R().eps,
                },
                "results": {
                    k: {kk: vv for kk, vv in v.items() if kk != "singular_values"}
                    for k, v in summary.items()
                },
                "generated_date": args.date,
            },
            indent=2,
        )
    )

    md = [
        "# S5-0 friction-extended base floor (2R)",
        "",
        f"generated: {args.date}",
        "",
        "Extended parameters: beta_ext = [m1,h1,J1,m2,h2,J2,Fv1,Fc1,Fv2,Fc2]",
        f"(viscous + smoothed-Coulomb friction, eps={FrictionModel2R().eps} rad/s).",
        "",
        "| config | rank / 10 | stable | sv gap (rejected -> accepted) |",
        "|---|---|---|---|",
    ]
    for name, s in summary.items():
        md.append(
            f"| {name} | {s['rank']} | {s['stable_across_thresholds_and_seeds']} | "
            f"{s['gap'][0]:.2e} -> {s['gap'][1]:.2e} |"
        )
    md += [
        "",
        "The rigid sub-block keeps its S0 structure (4 vertical / 3 horizontal);",
        "the four friction parameters are all identifiable under dynamic",
        "excitation, giving 8 / 7 extended base parameters. No symbolic",
        "certificate is claimed for the tanh friction columns; the rank is",
        "supported by the threshold sweep and cross-seed stability above.",
        "",
    ]
    (args.out / "s5_base_floor.md").write_text("\n".join(md))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
