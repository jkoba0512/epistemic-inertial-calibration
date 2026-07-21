# Vendored KUKA LBR iiwa14 model (S4 case study)

`iiwa14_no_collision.urdf` — a 7-DoF KUKA LBR iiwa14 model with full inertial
parameters and joint effort/velocity limits. It is vendored here so the project is
self-contained (no runtime dependency on `example-robot-data`). Pinocchio builds the
kinematic/dynamic model from the URDF alone; the visual mesh references in the URDF
are ignored by `pinocchio.buildModelFromUrdf` (model only, no geometry), so the mesh
files are intentionally NOT vendored.

## Provenance

- Obtained from RobotLocomotion/models, path `iiwa_description/urdf/iiwa14_no_collision.urdf`
  (https://github.com/RobotLocomotion/models).
- Those files were originally taken from the iiwa_stack project
  (https://github.com/SalvoVirga/iiwa_stack, SHA 2fa5bd5) and subsequently modified by
  Drake (mass/inertia and collision adjustments).

## License (redistributable, BSD)

All upstream license texts are reproduced verbatim in this directory. The two files
taken from RobotLocomotion/models (`iiwa_stack.LICENSE.txt` and
`DRAKE_iiwa_description.LICENSE.TXT`) are byte-identical to upstream.

- `iiwa_stack.LICENSE.txt` — as shipped upstream, this file carries **two** copyright
  blocks:
  - Copyright (c) 2015, Robert Krug & Todor Stoyanov, AASS Research Center, Örebro
    University, Sweden — **BSD-2-Clause** (no non-endorsement clause).
  - Copyright (c) 2016, Salvatore Virga and Marco Esposito, Technische Universität
    München — **BSD-3-Clause**.
- `DRAKE.LICENSE.TXT` — Drake's license, covering the Drake modifications to the
  model. Copyright 2012-2025 Robot Locomotion Group @ CSAIL — **BSD-3-Clause**.
- `DRAKE_iiwa_description.LICENSE.TXT` — the provenance note shipped upstream as
  `iiwa_description/LICENSE.TXT`. It is *not* a license text: it records the
  iiwa_stack origin and points to Drake's root `LICENSE.TXT`, which is vendored here
  as `DRAKE.LICENSE.TXT` so that this repository is self-contained.

These BSD licenses permit redistribution provided the copyright notices and license
texts are retained, which is what this directory and the top-level `NOTICE` do.

## Notes

- nq = nv = 7 (revolute joints); njoints = 8 (universe + 7).
- Joint effort limits (URDF): [320, 320, 176, 176, 110, 40, 40] N·m.
- Loaded via `iiwa_model.load_iiwa()`.
