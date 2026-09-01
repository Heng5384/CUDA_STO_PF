# Initial projection and moment audit

Projection status: `PASS_INITIAL_PSD_PROJECTION_CONSERVATION`. Production observables use the fixed-pivot measure; cell-integrated values are a reconstruction diagnostic only.

| Bins | M0 rel. 1600 | M3 rel. 1600 | β inventory rel. 1600 | M1 projection error | M2 projection error | Sv projection error | cell/M3 midpoint diff. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100.0 | 0 | 0 | 1.9484e-16 | 0.000683408 | 0.000687367 | 0.000687367 | 0.00179319 |
| 200.0 | 0 | 0 | 0 | 0.000146755 | 0.00014951 | 0.00014951 | 0.000448133 |
| 400.0 | 0 | 1.51837e-16 | 1.9484e-16 | 2.8869e-05 | 2.88468e-05 | 2.88468e-05 | 0.000112023 |
| 800.0 | 3.09237e-16 | 0 | 1.9484e-16 | 6.44232e-06 | 6.45401e-06 | 6.45401e-06 | 2.80051e-05 |
| 1600.0 | 0 | 0 | 0 | 1.41376e-06 | 1.40442e-06 | 1.40442e-06 | 7.00124e-06 |

Each discrete source radius is split non-negatively between bracketing fixed pivots to preserve M0 and M3. M1/M2/Sv projection error is reported in `initial_grid_moments.csv`.
