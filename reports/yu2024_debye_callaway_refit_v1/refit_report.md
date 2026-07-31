# Yu 2024 minimal refit

## Scope

This is a user-authorized refit to the two dashed Callaway curves in main-text
Figure 6b. It does not replace the published-parameter reproduction. The upper
curve is labelled **Annealed** (653 K for 48 h); the lower curve is
**As-quenched**.

All Yu parameters are frozen except:

1. the shared `A_N`;
2. one multiplier on the Annealed dislocation density, equivalently scaling
   both S11 and S13 because both rates are linear in `N_D`.

## Fitted values

| Parameter | Published | Fitted |
|---|---:|---:|
| A_N | 1.5 | 1.5107116 |
| Annealed dislocation-rate scale | 1.0 | 0.117276776 |
| Annealed N_D (m^-2) | 3.5e+15 | 4.10468716e+14 |
| Annealed N_D (cm^-2) | 3.5e+11 | 4.10468716e+10 |

The fitted effective dislocation density is 11.728% of the
published value. This is an effective curve-fit result, not a replacement
measurement.

## Accuracy against Yu Figure 6b

| State | MAPE | Maximum relative error |
|---|---:|---:|
| As-quenched | 0.233% | 0.549% |
| Annealed | 0.403% | 0.629% |

## Fit diagnostics

- observations: 14
- fitted parameters: 2
- degrees of freedom: 12
- chi-square using the digitization uncertainty: 3.2375
- reduced chi-square: 0.269791
- Jacobian condition number: 12.5456
- PF data used: `false`
- Sheskin data used: `false`

The formal intervals in `fitted_parameter_contract.json` are local
least-squares intervals only. Digitized points along the same drawn curve are
correlated, so those intervals must not be interpreted as physical confidence
bounds.

## Scientific interpretation

The shared intrinsic factor changes only slightly from 1.5 to
1.510712. Nearly all of the original Annealed mismatch is absorbed by
reducing the effective dislocation strength to 0.117277 of the
published S11/S13 value. This confirms that the discrepancy is localized to
the quantitative dislocation contract, but it does not identify whether the
unreported difference lies in `N_D`, Eq. S13 implementation, `gamma_prime`,
or another prefactor.
