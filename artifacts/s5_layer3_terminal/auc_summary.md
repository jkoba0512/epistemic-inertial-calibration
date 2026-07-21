# S5 Layer-3 AUC summary (higher score predicts failure)

generated: 2026-07-21 | N_SEEDS=50 | runs=5400 | failures=4865 | velocity faults=3600

| score | AUC | 95% CI |
|---|---|---|
| feasibility_risk_ext | 0.993 | [0.991, 0.995] |
| feasibility_risk_ext_est | 0.993 | [0.992, 0.995] |
| feasibility_risk_tau_only | 0.991 | [0.989, 0.993] |
| calibration_alpha_rmse | 0.549 | [0.525, 0.574] |
| holdout_torque_rmse | 0.556 | [0.533, 0.580] |

## AUC differences (95% CI)

| comparison | diff | 95% CI |
|---|---|---|
| est_minus_calibration | 0.444 | [0.419, 0.468] |
| est_minus_tau_only | 0.002 | [0.001, 0.004] |
| oracle_minus_est | -0.000 | [-0.001, -0.000] |

## Go/no-go decision at rho = 1

| score | missed failures | miss rate | false alarms |
|---|---|---|---|
| feasibility_risk_ext | 65 | 0.013 | 0 |
| feasibility_risk_ext_est | 65 | 0.013 | 0 |
| feasibility_risk_tau_only | 462 | 0.095 | 0 |

## Stratified calibration AUC (by online rho)

- rho in [None, 1.0]: n=600, failure_rate=0.11, calibration AUC = 0.910 [0.887, 0.932]
- rho in [1.0, 3.0]: n=1191, failure_rate=1.00, calibration AUC = n/a
- rho in [3.0, None]: n=3609, failure_rate=1.00, calibration AUC = n/a


## Separate null-space invariance control

- paired runs: 150
- outcome mismatches: 0
- maximum |delta rho_online|: 2.665e-15
- maximum holdout-torque RMSE change: 3.281e-15