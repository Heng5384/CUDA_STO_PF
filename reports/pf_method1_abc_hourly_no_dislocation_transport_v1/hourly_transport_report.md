# Method-1 A/B/C hourly full-PSD no-dislocation Yu transport

## Scope

This read-only calculation consumes all 45 hourly complete connected-component
PSDs from each of the three accepted 246^3 Method-1 production authorities
(135 PF snapshots total).  A merged connected component is counted once as the
physical scatterer present at that snapshot; historical lineage members are
not double counted.  No PF field, source term, or production output was changed.

The frozen host is the Yu 48 h non-particle baseline with `A_N=1.5`, S11/S13
dislocation scattering set to zero, and no Yu refit scale.  `fixed_6h_matrix`
isolates PSD evolution; `pf_time_varying_matrix` additionally passes each PF
far-field Ag value to point-defect scattering.  The direct complete PSD sum is
the reference at seven temperatures from 300 to 600 K.

## Descriptor sufficiency: fixed 6 h matrix

| Descriptor | absolute kappa MAPE | maximum kappa error | delta-kappa signal NRMSE | max abs delta error (W/mK) | exact A/B/C delta ordering |
|---|---:|---:|---:|---:|---:|
| Nv_plus_mean_R | 0.349280% | 1.293819% | 291.296% | 0.0275039 | 0.315 |
| Nv_plus_mean_R_plus_CV_lognormal | 0.037118% | 0.173409% | 42.942% | 0.00395087 | 0.260 |
| Sv_geometric_limit | 3.896996% | 8.793786% | 3223.113% | 0.145851 | 0.455 |
| Sv_plus_M6 | 0.095118% | 0.284063% | 63.220% | 0.00544575 | 0.256 |
| full_PSD | 0.000000% | 0.000000% | 0.000% | 0 | 1.000 |

## Descriptor sufficiency: PF time-varying matrix

| Descriptor | absolute kappa MAPE | maximum kappa error | delta-kappa signal NRMSE | max abs delta error (W/mK) | exact A/B/C delta ordering |
|---|---:|---:|---:|---:|---:|
| Nv_plus_mean_R | 0.350627% | 1.296696% | 52.637% | 0.0274889 | 0.442 |
| Nv_plus_mean_R_plus_CV_lognormal | 0.037324% | 0.174152% | 7.772% | 0.00395094 | 0.708 |
| Sv_geometric_limit | 3.918073% | 8.793786% | 583.371% | 0.145851 | 0.643 |
| Sv_plus_M6 | 0.095526% | 0.285398% | 11.427% | 0.00544315 | 0.510 |
| full_PSD | 0.000000% | 0.000000% | 0.000% | 0 | 1.000 |

Absolute-kappa error is diluted by the common host background, so the
delta-kappa signal NRMSE is the stricter measure of whether a compressed
descriptor preserves the microstructure-evolution signal.  The
`Nv_plus_mean_R_plus_CV_lognormal` result is conditional on the explicitly
registered lognormal closure; `Nv`, mean radius, and CV do not define a unique
PSD by themselves.

Best compressed absolute-kappa model: `Nv_plus_mean_R_plus_CV_lognormal`.
Best compressed evolution-signal model: `Nv_plus_mean_R_plus_CV_lognormal`.
The complete PSD remains the production reference regardless of compression
ranking.

## 48 h complete-PSD ensemble

| T (K) | fixed 6 h matrix kappa (W/mK) | PF matrix kappa (W/mK) | PF matrix delta from 6 h (W/mK) |
|---:|---:|---:|---:|
| 300 | 2.284604 ± 0.001512 | 2.279808 ± 0.006337 | -0.006662 ± 0.006330 |
| 400 | 1.786014 ± 0.000623 | 1.783033 ± 0.003625 | -0.005061 ± 0.003620 |
| 600 | 1.243672 ± 0.000140 | 1.242200 ± 0.001616 | -0.002991 ± 0.001614 |

## Qualification and boundary

- Hourly source snapshots: 135 (45 each for A/B/C).
- Transport prediction cells: 9450.
- Full-PSD vector/direct-sum maximum relative difference: 4.775e-16.
- Lognormal mean closure maximum: 3.468e-15.
- Lognormal CV closure maximum: 9.159e-16.
- Recomputed six-time full-PSD maximum relative difference from the existing
  registered production transport: 3.591e-16.
- Registered six-time PSD hashes identical: true.

This qualifies a conditional no-dislocation transport interface and descriptor
information-loss comparison.  It does not reproduce absolute experimental
thermal conductivity and does not add a PF-predicted dislocation density.

```text
status=PASS_PF_METHOD1_ABC_HOURLY_FULL_PSD_NO_DISLOCATION_DESCRIPTOR_SUFFICIENCY_V1
source_authority_status=PASS_EXPERIMENT_MATRIX_ANCHORED_A_B_C_COMPLETE_PASS_AUTHORITIES
source_snapshot_count=135
prediction_cell_count=9450
full_PSD_reference=true
dislocation_mode=DISLOCATION_OFF
yu_refit_scale_used=false
production_PF_modified=false
absolute_experimental_kappa_claim=false
```
