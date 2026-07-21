# S5 Layer-2 closed-loop comparison (friction 4R plant)

generated: 2026-07-19 | N_SEEDS=50 | extended base rank 16

| policy | logdet_cov | alpha_rmse | holdout_rmse | ee_err_max | faults |
|---|---|---|---|---|---|
| free_reference | -122.66 | 6.499e-02 | 3.944e-01 | nan | 0 |
| no_exploration | -88.22 | 1.125e-01 | 1.519e+00 | 0.0192 | 0 |
| smooth_random_nullspace | -78.18 | 1.125e-01 | 9.460e-01 | 0.0198 | 0 |
| fourier_nullspace | -126.14 | 4.881e-02 | 2.597e-01 | 0.0255 | 0 |

Gap fractions (share of the free-reference improvement recovered):

- logdet_cov / smooth_random_nullspace: -0.291
- logdet_cov / fourier_nullspace: 1.101
- holdout_torque_rmse / smooth_random_nullspace: 0.509
- holdout_torque_rmse / fourier_nullspace: 1.120
