# S5 closed-loop Layer-1 comparison - summary

generated: 2026-07-19
reference: 20 x 0.6 s segments at dt=0.002
envelope: |q|<=3.142, |qd|<=2.0, |qdd|<=4.0, nominal |tau|<=0.8*40.0
sensors: sigma_q=0.0001, sigma_qd=0.001, sigma_tau=0.01; qdd estimated from measured qd

All policies are continuous executable trajectories tracked closed-loop
on the true friction plant; identification uses measured data only.

## vertical

| rank | policy | alpha_rmse | logdet_cov | holdout_rmse | validation_rmse | faults |
|---|---|---|---|---|---|---|
| 1 | fourier_envelope | 4.594e-03 | -105.63 | 1.932e-02 | 8.235e-02 | 0 |
| 2 | smooth_random | 3.210e-02 | -101.46 | 6.803e-02 | 7.814e-02 | 0 |
| 3 | segment_active_ig | 3.782e-02 | -104.26 | 7.381e-02 | 7.812e-02 | 0 |
| 4 | segment_fim_greedy | 3.889e-02 | -103.53 | 7.459e-02 | 7.817e-02 | 0 |
| 5 | hold_sequence | 1.667e-01 | -90.46 | 3.629e-01 | 1.155e-01 | 0 |

Executability (means over seeds):

| policy | tracking_rmse | q_max | qd_max | tau_peak | saturation |
|---|---|---|---|---|---|
| hold_sequence | 0.016 | 0.43 | 0.39 | 26.2 | 0.000 |
| smooth_random | 0.015 | 1.05 | 0.83 | 25.0 | 0.000 |
| fourier_envelope | 0.013 | 2.81 | 1.83 | 21.3 | 0.000 |
| segment_fim_greedy | 0.016 | 1.02 | 0.98 | 27.1 | 0.000 |
| segment_active_ig | 0.016 | 1.03 | 0.99 | 27.4 | 0.000 |

## horizontal

| rank | policy | alpha_rmse | logdet_cov | holdout_rmse | validation_rmse | faults |
|---|---|---|---|---|---|---|
| 1 | fourier_envelope | 4.240e-03 | -85.61 | 1.867e-02 | 8.260e-02 | 0 |
| 2 | smooth_random | 3.382e-02 | -81.04 | 6.670e-02 | 7.802e-02 | 0 |
| 3 | segment_active_ig | 3.948e-02 | -84.85 | 7.127e-02 | 7.802e-02 | 0 |
| 4 | segment_fim_greedy | 4.026e-02 | -84.46 | 7.258e-02 | 7.809e-02 | 0 |
| 5 | hold_sequence | 1.793e-01 | -71.25 | 3.653e-01 | 1.157e-01 | 0 |

Executability (means over seeds):

| policy | tracking_rmse | q_max | qd_max | tau_peak | saturation |
|---|---|---|---|---|---|
| hold_sequence | 0.001 | 0.43 | 0.39 | 9.8 | 0.000 |
| smooth_random | 0.003 | 1.04 | 0.82 | 9.6 | 0.000 |
| fourier_envelope | 0.004 | 2.81 | 1.83 | 5.9 | 0.000 |
| segment_fim_greedy | 0.004 | 1.31 | 0.96 | 11.0 | 0.000 |
| segment_active_ig | 0.004 | 1.32 | 0.98 | 10.9 | 0.000 |
