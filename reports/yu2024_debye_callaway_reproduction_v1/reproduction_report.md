# Yu 2024 AQ/48 h reproduction report

## Result

AQ reproduces Yu's published model curve. The 48 h state does not reproduce it when all published equations and Table S2 parameters are applied literally. The final status is:

`BLOCKED_YU2024_MODEL_REPRODUCTION_MISMATCH`

No parameter was refitted, and the failure is not caused by numerical integration.

## Comparison with the published Yu model

| State | MAPE | Maximum relative error | RMSE (W m^-1 K^-1) | Gate |
|---|---:|---:|---:|---|
| AQ | 0.545% | 1.011% | 0.00910 | PASS |
| 48 h | 43.348% | 48.640% | 0.72158 | FAIL |

The required gates are MAPE at most 5% and maximum relative error at most 10%.

The digitized Yu-curve uncertainty is approximately `0.01127 W m^-1 K^-1`, far smaller than the 48 h residual of roughly `0.47-0.97 W m^-1 K^-1`.

## Comparison with SI Table S1 experiments

| State | MAPE | Maximum relative error | RMSE (W m^-1 K^-1) |
|---|---:|---:|---:|
| AQ | 1.205% | 3.001% | 0.01992 |
| 48 h | 41.740% | 49.666% | 0.70344 |

The priority gate is reproduction of the Yu model rather than refitting experiment. AQ also agrees closely with experiment, while the 48 h mismatch is already present relative to Yu's dashed curve.

## Numerical verification

- 256/512-point Gauss-Legendre maximum relative difference: `6.62e-14`
- Gauss-Legendre/adaptive-quadrature maximum relative difference: `6.81e-14`
- all full rates finite and positive on quadrature nodes: PASS
- all conductivities finite and positive: PASS
- low-frequency powers for phonon-phonon, point defect, and Rayleigh precipitate rates: PASS
- high-frequency geometric precipitate limit: PASS
- Rayleigh/geometric crossover continuity: PASS
- independently derived `gamma_prime`: `2.61778969265` versus published `2.62`, relative difference `0.0844%`
- two separate process runs: all generated file hashes identical

## Localization of the 48 h mismatch

At 303.94 K:

- full published S13 calculation: `1.0268 W m^-1 K^-1`
- same published inputs with dislocations disabled: `2.4533 W m^-1 K^-1`
- Yu Figure 6b digitized model: `1.9993 +/- 0.0113 W m^-1 K^-1`
- SI Table S1 experiment: `2.04 W m^-1 K^-1`

The discrepancy is therefore localized to the quantitative application of the dislocation contribution, particularly the Ag-decorated strain-field term in SI Eq. S13. The AQ calculation, which has `N_D = 0`, reproduces Yu and exercises the same intrinsic, boundary, point-defect, and precipitate equations.

SI Eq. S12 is the generic undecorated-dislocation expression; Eq. S13 explicitly says it *should be modified* with `gamma_prime` for impurity segregation. As a diagnostic only, using Eq. S12 instead of the required Eq. S13 gives an 8.04% MAPE and 9.37% maximum error against the 48 h Yu curve. This still fails the 5% MAPE gate and contradicts the SI instruction/Table S2 `gamma_prime`. It was not adopted.

Disabling dislocations gives a 17.43% MAPE and 22.71% maximum error. It was not adopted.

No undocumented scale factor, density conversion change, alternate `gamma_prime`, or parameter optimization was used.

## Why this is not a PASS

The source set is present, the equations can be evaluated, and the computation is stable and deterministic. However, both state reproductions must pass. Because the 48 h published-parameter result misses Yu's published model far beyond digitization uncertainty and the allowed gate, the requested PASS marker would be scientifically false.

## Recommended next action

Request the authors' original Debye-Callaway calculation worksheet/code or clarification of the exact 48 h dislocation-rate inputs used to draw Figure 6b/6d. Specifically confirm:

1. the numerical `N_D` value and unit passed to Eqs. S11/S13;
2. whether Figure 6 used Eq. S12 or S13;
3. whether `gamma_prime = 2.62` was added to `gamma`, substituted for it, or omitted in the plotted calculation;
4. the exact bracket implementation in Eq. S13;
5. any unreported unit conversion or prefactor.

Until that contract is supplied, the 48 h curve must remain blocked rather than refitted.
