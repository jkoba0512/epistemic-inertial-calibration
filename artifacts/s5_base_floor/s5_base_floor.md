# S5-0 friction-extended base floor (2R)

generated: 2026-07-19

Extended parameters: beta_ext = [m1,h1,J1,m2,h2,J2,Fv1,Fc1,Fv2,Fc2]
(viscous + smoothed-Coulomb friction, eps=0.05 rad/s).

| config | rank / 10 | stable | sv gap (rejected -> accepted) |
|---|---|---|---|
| vertical | 8 | True | 1.05e-13 -> 1.19e+01 |
| horizontal | 7 | True | 3.66e-15 -> 1.19e+01 |

The rigid sub-block keeps its S0 structure (4 vertical / 3 horizontal);
the four friction parameters are all identifiable under dynamic
excitation, giving 8 / 7 extended base parameters. No symbolic
certificate is claimed for the tanh friction columns; the rank is
supported by the threshold sweep and cross-seed stability above.
