# S5 iiwa Layer-3 (velocity-aware terminal feasibility)

generated: 2026-07-21 | N_SEEDS=50 | runs=2400 | failures=2000 | velocity faults=2000

| score | AUC | 95% CI |
|---|---|---|
| feasibility_risk_ext | 1.000 | [1.000, 1.000] |
| feasibility_risk_ext_est | 1.000 | [1.000, 1.000] |
| feasibility_risk_tau_only | 0.900 | [0.887, 0.912] |
| calibration_alpha_rmse | 0.500 | [0.468, 0.531] |

## Go/no-go at rho = 1

| score | missed failures | miss rate | false alarms |
|---|---|---|---|
| feasibility_risk_ext | 0 | 0.000 | 0 |
| feasibility_risk_ext_est | 0 | 0.000 | 0 |
| feasibility_risk_tau_only | 1000 | 0.500 | 0 |
