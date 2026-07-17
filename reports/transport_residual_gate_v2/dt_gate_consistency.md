# dt and gate consistency

The unchanged V1 matrix remains the all-gate/all-dt consistency dataset. Its row-wise `eta_R` is a rigorous upper bound on V2 `eta_global` by the triangle inequality. Exact interface and material-cell normalized metrics are evaluated only for the strict reference and at most two selected candidates, because the original queue did not record per-cell A_i and the V2 contract forbids a post-hoc eight-case rerun.

## Common-window gate ladder at dt8

| Gate | eta_R upper bound | max abs local D | signed sum D |
|---|---:|---:|---:|
| G12 | 3.218259e-12 | 4.874473e-14 | 4.145774e-14 |
| G10 | 2.029573e-10 | 1.559885e-11 | 4.707224e-14 |
| G9 | 7.652049e-09 | 2.363712e-10 | 5.235461e-14 |
| G8 | 6.772912e-08 | 3.982923e-09 | 1.681037e-13 |

Gate-tightening consistency (G8 -> G9 -> G10 -> G12 does not worsen the upper bound): `PASS`.

## Common-window G10 dt ladder

| dt | eta_R upper bound | max abs local D |
|---|---:|---:|
| dt16 | 1.840770e-10 | 4.013164e-11 |
| dt8 | 2.029573e-10 | 1.559885e-11 |
| dt4 | 3.769441e-10 | 3.084568e-11 |
| dt2 | 2.755865e-10 | 2.053490e-11 |

The G10 upper bound is not strictly monotone with dt, so no truncation-order claim is made from residual localization alone. Its maximum is `3.769441e-10`, however, and the pre-existing equal-time QoI refinement remains the authoritative trajectory check. The normalized upper-bound plateau is classified as `NEGLIGIBLE_AND_QOI_CONTROLLED`.

Selected replay/holdout rows and their exact normalized metrics are in `window_metrics.csv`; all original matrix QoIs remain in the immutable V1 reports.
