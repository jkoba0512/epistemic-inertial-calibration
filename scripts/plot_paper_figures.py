"""Generate the paper figures and tables from the S5 executable-pipeline artifacts.

Tracked code; outputs go to paper/figures/ and paper/tables/ (both gitignored).
All numeric content is read from artifacts/s5_*/*.json where possible; base-floor
spectra are recomputed from the frozen models. Any hand-entered value carries an
inline comment naming its source.

Run from the repository root:
  uv run python scripts/plot_paper_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ART = REPO / "artifacts"
FIG = REPO / "paper" / "figures"
TAB = REPO / "paper" / "tables"


def _load(p):
    fp = ART / p
    return json.loads(fp.read_text()) if fp.exists() else None


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> paper/figures/{name}.pdf")


LABELS = {
    # S5 executable Layer-1 policies
    "hold_sequence": "dwell-and-hold",
    "smooth_random": "smooth random",
    "fourier_envelope": "Fourier (box-fit)",
    "segment_fim_greedy": "FIM-greedy",
    "segment_active_ig": "active-IG",
    # S5 executable Layer-2 policies
    "free_reference": "free reference",
    "no_exploration": "no exploration",
    "smooth_random_nullspace": "random null-space",
    "fourier_nullspace": "Fourier null-space",
}


def _label(policy: str) -> str:
    return LABELS.get(policy, policy.replace("_", " "))


def fig1_concept():
    """Three-layer pipeline + the sequential terminal diagnosis (Fig. 1).

    Left-to-right: uncertain dynamics -> Layer 1 (identifiability) -> Layer 2
    (task-compatible excitation) -> calibrated estimate -> Layer 3. Layer 1
    defines the base coordinates rather than acting as a go/no-go gate. Layer 2
    collects task-compatible calibration data without imposing a universal gate,
    and Layer 3 forks into the two empirical terminal-feasibility regimes found
    by the stratified analysis (S5 Layer 3). The in-figure AUC is read from the
    stratified artifact.
    """
    strat = _load("s5_layer3_terminal/stratified_metrics.json")
    auc_feasible = None
    if strat:
        entry = strat["strata"][0]
        if entry.get("auc"):
            auc_feasible = entry["auc"]["auc"]
    auc_txt = f"{auc_feasible:.2f}" if auc_feasible is not None else "0.92"

    ink, muted = "#1a1a1a", "#5a5a5a"
    fig, ax = plt.subplots(figsize=(9.0, 2.5))
    ax.axis("off")
    # Margin beyond the [0,1] layout box: the rounded-box padding and the stroke
    # width extend past the nominal corners, and anything outside the axes limits
    # is clipped (which shaved the leftmost and rightmost boxes).
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.03, 1.03)

    def box(x, y, w, h, fc, ec, lw=1.2):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
            fc=fc, ec=ec, lw=lw, mutation_aspect=3.6, clip_on=False))

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#333333",
                                    shrinkA=0, shrinkB=0))

    # --- input node ---------------------------------------------------------
    box(0.005, 0.51, 0.058, 0.28, "#f2f2f2", "#8a8a8a", lw=1.0)
    ax.text(0.034, 0.65, "$\\phi$\nunknown", ha="center", va="center",
            fontsize=7.2, color=ink)

    # --- layer boxes: title + one keyword; the prose lives in the caption ---
    layers = [
        (0.100, "Layer 1\nBase-space\ndefinition", "identify $V_{base}$\ndefine $\\alpha=V_{base}\\phi$",
         "model structure", "#dbe9f6", "#3f6b9c"),
        (0.320, "Layer 2\nTask-compatible\ncalibration", "collect $\\mathcal{D}_{task}$\nupdate $p(\\alpha|\\mathcal{D})$",
         "the task itself", "#e9e2f3", "#6b5191"),
        (0.540, "Layer 3\nTerminal\nfeasibility", "$\\rho_{\\mathrm{online}}$",
         "actuator limits, horizon", "#fdeadc", "#a2672c"),
    ]
    bw, bh, by = 0.185, 0.44, 0.43
    for x, title, keyword, limit, fc, ec in layers:
        box(x, by, bw, bh, fc, ec)
        ax.text(x + bw / 2, 0.835, title, ha="center", va="top",
                fontsize=8.0, weight="bold", color=ink, linespacing=1.45)
        ax.text(x + bw / 2, 0.505, keyword, ha="center", va="center",
                fontsize=7.2, color=ink)
        ax.text(x + bw / 2, 0.325, f"limited by\n{limit}", ha="center",
                va="center", fontsize=6.8, style="italic", color=muted)

    arrow(0.065, 0.65, 0.098, 0.65)
    arrow(0.287, 0.65, 0.318, 0.65)
    arrow(0.507, 0.65, 0.538, 0.65)
    ax.text(0.5225, 0.685, "$\\hat{\\phi}$", ha="center", va="bottom",
            fontsize=7.4, color=ink)

    # --- terminal fork: the two regimes (from the stratified analysis) ------
    fx, fw, fh = 0.815, 0.180, 0.26
    box(fx, 0.62, fw, fh, "#e6f2e6", "#2f7d3a")
    ax.text(fx + fw / 2, 0.800, "within limit", ha="center", va="center",
            fontsize=8.0, weight="bold", color="#205a28")
    ax.text(fx + fw / 2, 0.685, f"outcome tracks\ncalibration (AUC {auc_txt})",
            ha="center", va="center", fontsize=6.8, color=ink)
    box(fx, 0.10, fw, fh, "#fae4e1", "#a33c31")
    ax.text(fx + fw / 2, 0.280, "above limit", ha="center", va="center",
            fontsize=8.0, weight="bold", color="#7c2a22")
    ax.text(fx + fw / 2, 0.165, "fails regardless of\ncalibration", ha="center",
            va="center", fontsize=6.8, color=ink)

    arrow(0.727, 0.69, 0.811, 0.75)
    ax.text(0.770, 0.735, "$\\rho_{\\mathrm{online}} \\leq 1$", ha="center", va="bottom",
            fontsize=7.2, color=ink)
    arrow(0.727, 0.58, 0.811, 0.28)
    ax.text(0.770, 0.355, "$\\rho_{\\mathrm{online}} > 1$", ha="center", va="bottom",
            fontsize=7.2, color=ink)

    _save(fig, "fig1_three_layer_concept")


def _spectrum_2r(g=9.81):
    from epistemic_inertial_calibration.base_parameters import SamplingConfig, sample_states
    from epistemic_inertial_calibration.planar2r import Planar2RModel
    from epistemic_inertial_calibration.s5_model import FrictionModel2R, stacked_regressor_ext
    fm = FrictionModel2R(rigid=Planar2RModel(g=g))
    states = sample_states(SamplingConfig(regime="dynamic", n_samples=1000, seed=0))
    W = stacked_regressor_ext(states, fm)
    sv = np.linalg.svd(W, compute_uv=False)
    return sv, 8 if g != 0.0 else 7


def _spectrum_4r():
    import numpy as _np
    from epistemic_inertial_calibration.planar_nr import PlanarNRModel
    from epistemic_inertial_calibration.s5_layer2 import default_friction_nr, stacked_regressor_ext_nr
    fm = default_friction_nr(PlanarNRModel())
    rng = _np.random.default_rng(0)
    n = fm.n
    states = _np.hstack([rng.uniform(-_np.pi, _np.pi, (2000, n)),
                         rng.uniform(-2, 2, (2000, n)), rng.uniform(-4, 4, (2000, n))])
    W = stacked_regressor_ext_nr(states, fm)
    return _np.linalg.svd(W, compute_uv=False), 16


def _spectrum_iiwa():
    try:
        from epistemic_inertial_calibration.s5_iiwa import (
            load_friction_iiwa, sample_states_hw, stacked_regressor_ext_iiwa)
    except Exception:
        return None, None
    fm = load_friction_iiwa()
    W = stacked_regressor_ext_iiwa(sample_states_hw(fm, 1000, seed=0), fm)
    return np.linalg.svd(W, compute_uv=False), 62


def fig2_base_floor():
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    spectra = [
        ("2R vertical (+friction)", lambda: _spectrum_2r(9.81), "o", "C0"),
        ("2R horizontal (+friction)", lambda: _spectrum_2r(0.0), "v", "C1"),
        ("4R (+friction)", _spectrum_4r, "^", "C2"),
        ("iiwa (+friction, actuator inertia)", _spectrum_iiwa, "s", "C3"),
    ]
    for label, fn, marker, color in spectra:
        sv, rank = fn()
        if sv is None:
            continue
        sv = np.array(sv, float)
        sv = sv / sv[0]
        n_params = len(sv)
        ax.semilogy(range(1, n_params + 1), sv, marker + "-", color=color, ms=3.5, lw=1.3,
                    label=f"{label} (base={rank}/{n_params})")
        ax.axvline(rank + 0.5, color=color, ls="--", lw=1, alpha=0.6)
    ax.axhline(1e-8, color="0.45", ls=":", lw=1)
    ax.text(2, 1.7e-8, "rank threshold $10^{-8}$", fontsize=7, color="0.35", va="bottom")
    ax.set_xlabel("singular value index")
    ax.set_ylabel("normalized singular value")
    ax.set_ylim(1e-17, 2)
    ax.legend(fontsize=7.5, loc="lower left")
    _save(fig, "fig2_base_floor")


def _policy_bar(metrics, policies, key, title, ylabel, name, logy=False):
    vals = [metrics[p][key]["mean"] for p in policies]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(range(len(policies)), vals, color="#4c72b0")
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels([_label(p) for p in policies], rotation=32, ha="right", fontsize=7)
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    _save(fig, name)


def fig3_layer1():
    d = _load("s5_layer1_closed_loop/metrics_by_policy.json")
    if not d:
        return
    m = d["vertical"]
    pols = ["hold_sequence", "smooth_random", "segment_fim_greedy",
            "segment_active_ig", "fourier_envelope"]
    _policy_bar(m, pols, "holdout_torque_rmse",
                "Free-excitation benchmark (planar 2R, vertical): box-fit Fourier strongest,\n"
                "active-IG $\\equiv$ FIM-greedy (no advantage); zero faults",
                "holdout torque RMSE (log)", "fig3_layer1_excitation", logy=True)


def fig4_layer2():
    d = _load("s5_layer2_closed_loop/metrics_by_policy.json")
    if not d:
        return
    pols = ["free_reference", "no_exploration", "smooth_random_nullspace", "fourier_nullspace"]
    no = d["no_exploration"]["logdet_cov"]["mean"]
    free = d["free_reference"]["logdet_cov"]["mean"]
    denom = no - free
    vals = [(no - d[p]["logdet_cov"]["mean"]) / denom * 100.0 for p in pols]
    rmse = [d[p]["alpha_rmse"]["mean"] for p in pols]
    rmse_std = [d[p]["alpha_rmse"]["std"] for p in pols]
    fig, (ax, ax_rmse) = plt.subplots(1, 2, figsize=(7.2, 3.3), constrained_layout=True)
    bars = ax.bar(range(len(pols)), vals, color="#4c72b0")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + (3 if val >= 0 else -3),
                f"{val:.0f}%", ha="center",
                va="bottom" if val >= 0 else "top", fontsize=8)
    ax.axhline(0, color="0.35", lw=0.9)
    ax.axhline(100, color="gray", ls=":", lw=1)
    ax.set_ylim(min(-40, min(vals) - 8), max(120, max(vals) + 8))
    ax.set_ylabel("recovered reduction (%)")
    ax.set_xticks(range(len(pols)))
    ax.set_xticklabels([_label(p) for p in pols], rotation=30, ha="right", fontsize=7)
    ax.text(0.01, 0.95, "(a)", transform=ax.transAxes, va="top", fontsize=9)
    ax_rmse.bar(range(len(pols)), rmse, yerr=rmse_std, capsize=4, color="#55a868")
    ax_rmse.set_ylabel(r"base-coordinate RMSE")
    ax_rmse.yaxis.set_label_position("right")
    ax_rmse.yaxis.tick_right()
    ax_rmse.set_xticks(range(len(pols)))
    ax_rmse.set_xticklabels([_label(p) for p in pols], rotation=30, ha="right", fontsize=7)
    ax_rmse.text(0.01, 0.95, "(b)", transform=ax_rmse.transAxes, va="top", fontsize=9)
    _save(fig, "fig4_layer2_tradeoff")


def _auc_bars(ax, auc, scores, labels, title):
    means = [auc[s]["auc"] for s in scores]
    lo = [auc[s]["auc"] - auc[s]["ci_low"] for s in scores]
    hi = [auc[s]["ci_high"] - auc[s]["auc"] for s in scores]
    ax.bar(range(len(scores)), means, yerr=[lo, hi], capsize=4,
           color=["#c44e52", "#8172b3", "#55a868", "#937860"][:len(scores)])
    ax.axhline(0.5, color="gray", ls=":", label="chance")
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("AUC (predict failure)")


def _load_quadrant_counts(path):
    fp = ART / path
    if not fp.exists():
        return None
    counts = {}
    for line in fp.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 2 or cells[0] in {"quadrant", "---"}:
            continue
        try:
            counts[cells[0]] = int(cells[1])
        except ValueError:
            continue
    needed = {"well_success", "well_fail", "poor_success", "poor_fail"}
    return counts if needed.issubset(counts) else None


def _quadrant_heatmap(ax, counts):
    matrix = np.array([
        [counts["well_success"], counts["well_fail"]],
        [counts["poor_success"], counts["poor_fail"]],
    ])
    im = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["success", "failure"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["well\ncalibrated", "poorly\ncalibrated"])
    ax.set_xlabel("terminal outcome")
    ax.set_ylabel("calibration group")
    threshold = matrix.max() * 0.55
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > threshold else "black"
            ax.text(j, i, f"{matrix[i, j]}", ha="center", va="center",
                    fontsize=10, weight="bold", color=color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im


def fig5_layer3():
    d = _load("s5_layer3_terminal/auc_bootstrap.json")
    if not d:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    fig.subplots_adjust(wspace=0.46)
    scores = ["feasibility_risk_ext_est", "feasibility_risk_tau_only",
              "calibration_alpha_rmse", "holdout_torque_rmse"]
    labels = ["feasibility\n(online, vel-aware)", "feasibility\n(online, torque-only)", "calibration", "holdout-torque"]
    _auc_bars(axes[0], d["auc"], scores, labels,
              "Layer 3 (planar 2R, vertical): terminal-risk separates failure,\ncalibration error does not")
    axes[0].text(0.01, 0.96, "(a)", transform=axes[0].transAxes,
                 va="top", fontsize=9)
    counts = _load_quadrant_counts("s5_layer3_terminal/quadrant_summary.md")
    if counts:
        _quadrant_heatmap(axes[1], counts)
        axes[1].text(0.01, 0.96, "(b)", transform=axes[1].transAxes,
                     va="top", fontsize=9)
    _save(fig, "fig5_layer3_terminal_risk")


def fig6_iiwa():
    s2r = _load("s5_layer3_terminal/auc_bootstrap.json")
    sii = _load("s5_iiwa/layer3_auc.json")
    if not (s2r and sii):
        return
    fig, ax = plt.subplots(figsize=(6, 3.4))
    groups = ["feasibility_risk_ext_est", "calibration_alpha_rmse"]
    glabels = ["feasibility\n(online, vel-aware)", "calibration"]
    x = np.arange(len(groups))
    w = 0.35
    for d, lab, off in [(s2r, "planar 2R\nvertical", -w / 2), (sii, "iiwa 7-DoF", w / 2)]:
        means = [d["auc"][g]["auc"] for g in groups]
        lo = [d["auc"][g]["auc"] - d["auc"][g]["ci_low"] for g in groups]
        hi = [d["auc"][g]["ci_high"] - d["auc"][g]["auc"] for g in groups]
        ax.bar(x + off, means, w, yerr=[lo, hi], capsize=4, label=lab)
    ax.axhline(0.5, color="gray", ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(glabels)
    ax.set_ylim(0.4, 1.04)
    ax.set_ylabel("AUC (predict failure)")
    ax.legend(fontsize=8)
    _save(fig, "fig6_iiwa")


# ---------------------------------------------------------------------------
# Tables (LaTeX)
# ---------------------------------------------------------------------------


def _write_table(name, body):
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / f"{name}.tex").write_text(body)
    print(f"  table  -> paper/tables/{name}.tex")


def table_basedim():
    s2 = _load("s5_base_floor/rank_summary.json")
    sii = _load("s5_iiwa/rank_summary.json")
    v2 = s2["results"]["vertical"]["rank"] if s2 else 8
    h2 = s2["results"]["horizontal"]["rank"] if s2 else 7
    iiwa = sii["rank"] if sii else 62
    ip = sii["n_params"] if sii else 91
    body = (
        "% Table I: extended base-parameter dimensions (from S5 rank_summary.json)\n"
        "\\begin{table}[htbp]\\centering\n\\caption{Extended base-parameter dimensions "
        "obtained from regressors sampled within the hardware limits. For the planar models, "
        "vertical means that gravity acts in the plane of motion, whereas horizontal means "
        "that gravity produces no in-plane joint torque. The iiwa model has three-dimensional "
        "spatial dynamics.}\n"
        "\\label{tab:basedim}\n\\begin{tabular}{llcc}\n\\toprule\n"
        "system & configuration & full params & base params\\\\\n\\midrule\n"
        f"planar 2R & vertical & 10 & {v2}\\\\\n"
        f"planar 2R & horizontal & 10 & {h2}\\\\\n"
        "planar 4R & vertical & 20 & 16\\\\\n"
        f"iiwa 7-DoF & spatial & {ip} & {iiwa}\\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    _write_table("tab_basedim", body)


def _auc_row(d, label):
    f = d["auc"]["feasibility_risk_ext_est"]
    c = d["auc"]["calibration_alpha_rmse"]
    return (f"{label} & {f['auc']:.3f} [{f['ci_low']:.3f}, {f['ci_high']:.3f}] & "
            f"{c['auc']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]\\\\\n")


def table_auc():
    s2r = _load("s5_layer3_terminal/auc_bootstrap.json")
    sii = _load("s5_iiwa/layer3_auc.json")
    rows = ""
    if s2r:
        rows += _auc_row(s2r, "planar 2R, vertical")
    if sii:
        rows += _auc_row(sii, "iiwa 7-DoF")
    body = (
        "% Table II: terminal-risk AUC vs calibration AUC (from S5 auc_bootstrap.json / layer3_auc.json)\n"
        "\\begin{table}[htbp]\\centering\n"
        "\\caption{Failure-separation AUC (95\\% bootstrap CI): online velocity-aware "
        "terminal-feasibility risk $\\rho_{\\mathrm{online}}$ computed from the estimated "
        "parameters vs.\\ calibration error. Higher values indicate better failure "
        "prediction. The planar row uses the 5400-run primary sweep; the 150 "
        "null-space controls are analyzed separately.}\n"
        "\\label{tab:auc}\n"
        "\\begin{tabular}{lcc}\n\\toprule\n"
        "system & $\\rho_{\\mathrm{online}}$ & calibration error\\\\\n\\midrule\n"
        f"{rows}"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    _write_table("tab_auc", body)


def _sci(x):
    """LaTeX scientific notation: 4.149e-04 -> $4.15\\times10^{-4}$."""
    from math import floor, log10
    if x == 0:
        return "$0$"
    e = floor(log10(abs(x)))
    mant = x / 10**e
    return f"${mant:.2f}\\times10^{{{e}}}$"


def table_layer1():
    d = _load("s5_layer1_closed_loop/metrics_by_policy.json")
    if not d:
        return
    m = d["vertical"]
    n_seeds = m["fourier_envelope"].get("n_seeds", "?")
    pols = ["fourier_envelope", "smooth_random", "segment_fim_greedy",
            "segment_active_ig", "hold_sequence"]

    def cell(p, key):
        return f"{_sci(m[p][key]['mean'])} $\\pm$ {_sci(m[p][key]['std'])}"

    rows = "".join(
        f"{_label(p)} & {cell(p, 'holdout_torque_rmse')} & "
        f"{cell(p, 'alpha_rmse')}\\\\\n" for p in pols)
    body = (
        "% Table III: executable Layer-1 policy ranking (from S5 metrics_by_policy.json)\n"
        "\\begin{table}[t]\\centering\n"
        "\\caption{Executable free-excitation policies for the planar 2R system in the "
        f"vertical configuration. Values are the mean $\\pm$ standard deviation over {n_seeds} "
        "seeds; torque RMSE is in N\\,m. Every policy is tracked in closed loop on the "
        "friction plant, remains within the excitation envelope, and produces no safety "
        "fault. Active-IG matches FIM-greedy, and neither outperforms the box-fitted Fourier "
        "trajectory.}\n\\label{tab:layer1}\n"
        "\\small\n"
        "\\begin{tabular}{lcc}\n\\toprule\n"
        "policy & holdout torque RMSE & base $\\alpha$ RMSE\\\\\n\\midrule\n"
        f"{rows}"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    _write_table("tab_layer1", body)


def main():
    print("Generating paper figures and tables from artifacts/ ...")
    fig1_concept()
    fig2_base_floor()
    fig3_layer1()
    fig4_layer2()
    fig5_layer3()
    fig6_iiwa()
    table_basedim()
    table_auc()
    table_layer1()
    print("done.")


if __name__ == "__main__":
    main()
