# Artifact organization

The `s5_*` directories contain the lightweight summaries used by the final
reproducibility pipeline:

- `s5_base_floor`: planar 2R structural base-rank results;
- `s5_layer1_closed_loop`: executable free-excitation comparison;
- `s5_layer2_closed_loop`: planar 4R task-compatible excitation;
- `s5_layer3_terminal`: planar 2R terminal-feasibility analysis;
- `s5_iiwa`: seven-degree-of-freedom generalization.

Tracked JSON and Markdown files provide compact summaries. The final S5 NPZ
projections and per-run CSV tables listed below are also tracked so that
Zenodo's GitHub integration includes them in the archived release:

```text
artifacts/s5_base_floor/base_projection_ext_horizontal.npz
artifacts/s5_base_floor/base_projection_ext_vertical.npz
artifacts/s5_layer1_closed_loop/results_by_seed.csv
artifacts/s5_layer2_closed_loop/base_projection_ext_4r.npz
artifacts/s5_layer2_closed_loop/results_by_seed.csv
artifacts/s5_layer3_terminal/results_by_run.csv
artifacts/s5_iiwa/base_projection_ext.npz
artifacts/s5_iiwa/layer2_results_by_seed.csv
artifacts/s5_iiwa/layer3_results_by_run.csv
```

These files allow the reported aggregate metrics and AUC values to be checked
without rerunning the simulations. They remain reproducible from the scripts
and fixed configurations in this repository.
