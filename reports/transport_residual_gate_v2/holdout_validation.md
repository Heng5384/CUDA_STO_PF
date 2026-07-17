# Holdout validation

Extra main simulations: `2` (maximum allowed: 3).
The fifth equal-time window was not used to set V2 thresholds or the strict-reference floor.

| Case | eta global | eta interface | eta cell | b signed | b interface | QoI | Numerical | Holdout |
|---|---:|---:|---:|---:|---:|---|---|---|
| G12+dt32 | 2.508e-12 | 4.310e-12 | 2.171e-08 | 2.771e-03 | 2.052e-03 | PASS | PASS | FAIL |
| G9+dt4 | 8.987e-09 | 6.975e-09 | 2.086e-04 | 1.629e-07 | 3.431e-03 | PASS | PASS | FAIL |

Both holdouts pass the magnitude-normalized defect, QoI, and numerical contracts. They fail only the frozen signed-bias contract: the strict reference exceeds the global and interface signed-fraction limit, while G9+dt4 exceeds the interface signed-fraction limit.
