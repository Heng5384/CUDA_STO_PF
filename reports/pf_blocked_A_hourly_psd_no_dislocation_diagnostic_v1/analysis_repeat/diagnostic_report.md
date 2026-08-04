# Blocked Method-1 A: hourly full-PSD no-dislocation transport diagnostic

## Scope

This is a read-only diagnostic on the 45 copied overall resolved-particle PSD snapshots from Method-1 A.  The PF solver reached 48 h, but its particle-lineage gate is blocked; hence this is not a production authority, cannot enter the A/B/C ensemble, and makes no absolute experimental-kappa claim.  Particle IDs are not used as persistent identities: only each snapshot's radius multiset is used.

The frozen transport contract uses the Yu 48 h non-precipitate host, A_N=1.5, S11/S13=0, and no Yu refit scale.  `fixed_6h_matrix` isolates PSD evolution; `pf_time_varying_matrix` additionally passes raw A far-field xAg into point-defect scattering.

## Numerical checks

- PSD/observable descriptor closure maximum: 8.945e-16
- full-PSD vector/direct-sum maximum relative difference: 3.096e-16
- lognormal mean closure maximum: 2.827e-15
- lognormal CV closure maximum: 9.159e-16

## fixed_6h_matrix

| Descriptor | kappa MAPE | kappa maximum | delta-kappa NRMSE | max abs delta error (W/mK) |
|---|---:|---:|---:|---:|
| Nv_plus_mean_R | 0.336257% | 0.915682% | 265.478% | 0.0188329 |
| Nv_plus_mean_R_plus_CV_lognormal | 0.041636% | 0.173409% | 46.113% | 0.00395087 |
| Sv_geometric_limit | 3.833406% | 8.793753% | 3205.386% | 0.132474 |
| Sv_plus_M6 | 0.082650% | 0.193096% | 48.740% | 0.00336958 |
| full_PSD | 0.000000% | 0.000000% | 0.000% | 0 |

## pf_time_varying_matrix

| Descriptor | kappa MAPE | kappa maximum | delta-kappa NRMSE | max abs delta error (W/mK) |
|---|---:|---:|---:|---:|
| Nv_plus_mean_R | 0.337597% | 0.919265% | 50.331% | 0.0188105 |
| Nv_plus_mean_R_plus_CV_lognormal | 0.041857% | 0.174152% | 8.759% | 0.00395094 |
| Sv_geometric_limit | 3.853258% | 8.793753% | 608.875% | 0.132475 |
| Sv_plus_M6 | 0.083030% | 0.195105% | 9.239% | 0.00335915 |
| full_PSD | 0.000000% | 0.000000% | 0.000% | 0 |

## Boundary

These values qualify only the transport-interface calculation applied to raw A snapshot PSDs.  They neither repair the failed particle-lineage audit nor reproduce an absolute experimental thermal conductivity.

```text
status=PASS_PF_BLOCKED_A_HOURLY_PSD_NO_DISLOCATION_DIAGNOSTIC_V1
source_production_authority=false
source_root_status=BLOCKED_246CUBE_6H48H_PRODUCTION_DRIVER_V1
source_lineage_status=BLOCKED_PERIODIC_OVERLAP_PARTICLE_LINEAGE_V1
dislocation_mode=DISLOCATION_OFF
absolute_experimental_kappa_claim=false
```
