# Scattering-mechanism analysis

## Interpretation rule

For each mechanism group, `delta_kappa_off` is

\[
\Delta\kappa_{\mathrm{off},i}
=\kappa(\text{all except }i)-\kappa(\text{all}).
\]

It is a sensitivity measure. These values are not additive independent thermal resistances and must not be summed as a decomposition of `kappa`.

The reported effective rates are averages weighted by the full conductivity integration kernel. The dominant frequency interval is the 10%-90% cumulative-conductivity interval in `omega / 1e12 rad/s`.

## AQ

At 303.06 K:

| Cumulative model | kappa_lat (W m^-1 K^-1) |
|---|---:|
| combined Normal + Umklapp | 2.7163 |
| + grain boundary | 2.6549 |
| + point defect | 2.3419 |
| + small precipitates | 2.0801 |
| + big precipitates | 1.6560 |

The full AQ calculation has no dislocation scattering because Yu registers `N_D = 0`. At 303.06 K, the mechanism-off sensitivities are:

- boundary: `0.0333 W m^-1 K^-1`
- point defect: `0.1895 W m^-1 K^-1`
- all precipitates: `0.6859 W m^-1 K^-1`
- dislocations: `0`

The precipitate sensitivity is therefore the largest structural-defect sensitivity, consistent with Yu's narrative. Small and big populations both matter: their cumulative reductions at 303.06 K are approximately 0.262 and 0.424 W m^-1 K^-1, respectively. The 10%-90% kernel interval is `3.14-15.94` on the paper's angular-frequency axis.

At 573.15 K, the full value is `1.0612 W m^-1 K^-1`; the point-defect and precipitate sensitivities decrease to approximately `0.0724` and `0.2601 W m^-1 K^-1`.

## 48 h published-parameter calculation

At 303.94 K:

| Cumulative model | kappa_lat (W m^-1 K^-1) |
|---|---:|
| combined Normal + Umklapp | 2.7016 |
| + grain boundary | 2.6452 |
| + point defect | 2.4745 |
| + small precipitates | 2.4745 |
| + all precipitates | 2.4533 |
| + dislocation core and Ag-decorated strain field | 1.0268 |

The mechanism-off sensitivities are:

- boundary: `0.00047 W m^-1 K^-1`
- point defect: `0.0498 W m^-1 K^-1`
- all precipitates: `0.00074 W m^-1 K^-1`
- dislocations: `1.4265 W m^-1 K^-1`

The qualitative ranking agrees with Yu: coarsened precipitates are weak and the Ag-decorated dislocation contribution is the dominant defect sensitivity. The quantitative magnitude does not agree with Yu Figure 6b/6d. The published S13 input drives `kappa_lat` to `1.0268 W m^-1 K^-1`, whereas the digitized Yu model is `1.9993 +/- 0.0113 W m^-1 K^-1`.

At 572.27 K, the full calculation is `0.7487 W m^-1 K^-1`, and the dislocation-off sensitivity remains `0.6122 W m^-1 K^-1`.

## Dominant-rate wording

The kernel-weighted *total* rate is largest in the combined phonon-phonon term for both states at the tabulated temperatures. Calling dislocations "dominant" for 48 h refers to the dominant *defect-induced reduction*, not a larger rate than the intrinsic phonon-phonon rate. The output preserves both metrics to avoid conflating them.

## Normal/Umklapp limitation

The source supplies only the combined rate. Separate Normal and Umklapp budgets cannot be produced without adding an unauthorized model. Their status is therefore `COMBINED_NOT_SEPARABLE`.
