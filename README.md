# epistemic-inertial-calibration

Simulation code and reproducibility artifacts for active inertial
self-calibration of robot manipulators. The project separates calibration into
three diagnostic layers:

1. the structurally identifiable base-parameter subspace;
2. information available from task-compatible excitation; and
3. terminal task feasibility under torque, velocity, and horizon limits.

The current study covers planar 2R and 4R systems and a KUKA LBR iiwa 14 R820
seven-degree-of-freedom model. All reported experiments use continuous reference
trajectories, closed-loop tracking, friction-aware plants, and identification
from measured and differentiated states.

## Scope of this repository

The `S5` pipeline is the reproducibility path for the associated study. Earlier
development stages and internal interpretation notes are intentionally excluded
from the public repository.

The manuscript source and submission files are intentionally excluded from this
repository. Figure-generation code is included so that the reported plots and
tables can be reconstructed from the final artifacts.

## Installation

Python 3.11 or later and [uv](https://docs.astral.sh/uv/) are recommended.

```bash
git clone https://github.com/jkoba0512/epistemic-inertial-calibration.git
cd epistemic-inertial-calibration
uv sync
uv run pytest
uv run ruff check .
```

The iiwa experiments require the optional Pinocchio dependency:

```bash
uv sync --extra iiwa
```

See `USING_ENV.md` if the virtual environment must be stored outside the
repository.

## Reproducing the final experiments

The commands below regenerate the final `S5` artifacts. Full experiments use
fixed random seeds and may take substantially longer than the test suite.

### Layer 1: base subspace and executable free excitation

```bash
uv run python scripts/run_s5_base_floor.py
uv run python scripts/run_s5_layer1.py --full
```

These commands determine the friction-extended base subspaces of the planar 2R
models and compare executable calibration trajectories under a common budget.

### Layer 2: task-compatible excitation

```bash
uv run python scripts/run_s5_layer2.py --full
```

This command runs the planar 4R end-effector hold experiment with no
exploration, task-compatible null-space excitation, and an executable
free-excitation reference.

### Layer 3: terminal-feasibility diagnosis

```bash
uv run python scripts/run_s5_layer3.py
```

This command generates the 5,400-rollout primary planar sweep and the separate
150-run null-space invariance control. In the final primary analysis, the online
velocity-aware feasibility score predicts failure with AUC 0.993, compared with
0.549 for base-coordinate calibration error.

### Seven-degree-of-freedom generalization

```bash
uv run --extra iiwa python scripts/run_s5_iiwa.py --full
```

This command runs the extended base-rank analysis and Layers 2 and 3 on the
KUKA LBR iiwa model. The final Layer-3 sweep contains 2,400 rollouts.

### Figures and tables

```bash
uv run --extra iiwa python scripts/plot_paper_figures.py
```

The script reads the tracked `artifacts/s5_*` summaries and writes generated
files under the gitignored `paper/` directory. The manuscript itself is not
required to run the numerical experiments.

## Repository layout

```text
src/epistemic_inertial_calibration/  Models, estimators, policies, and experiments
scripts/run_s5_*.py                  Final experiment entry points
scripts/plot_paper_figures.py        Figure and table generation
artifacts/s5_*                       Final lightweight result summaries
tests/                               Unit and pipeline tests
assets/iiwa/                         Vendored iiwa URDF and upstream licenses
```

## Reproducibility data policy

Git tracks source code, configurations, fixed-seed summaries, final S5 per-run
tables, and the small base-projection files needed to inspect the archived
analysis directly. Legacy per-run data, generated figures, and manuscript files
remain excluded. Zenodo's GitHub integration archives the tagged GitHub release,
so the software and final research data receive one versioned release DOI. That
DOI will be recorded in `CITATION.cff` and in the associated article.

## License

This project is licensed under the Apache License, Version 2.0. The vendored
KUKA LBR iiwa URDF retains its upstream BSD licenses. See `NOTICE` and
`assets/iiwa/` for attribution and license texts.

## Citation

Citation metadata are provided in `CITATION.cff`. The software-release DOI and
article DOI will be added when they become available.
