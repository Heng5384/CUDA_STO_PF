# Real A/B/C 6--8 h descriptor sufficiency comparison

## Scope and data authority

This read-only analysis uses all 51 real short-screening snapshots (17 per
replicate) from `pf_246cube_three_seed_library_handoff_v1`.  A, B, and C share
the same 246^3 box, 380 C physics, elastic contract, `dt_code=0.02`, and
6--8 h time grid.  All three short screenings PASS; the one C diffuse-tail
merge has the frozen merge-aware PASS.  Only snapshot PSDs are used, so no
new independent post-merge identity claim is made.

The source fixture family has a 6 h far-field Ag fraction near 0.006788, not
the later experiment-anchored 0.0062 Method-1 fixture.  Therefore this report
tests descriptor information loss only.  It is not an APT concentration-fit
or absolute experimental thermal-conductivity result.

## Mathematical closures

- `Nv_plus_mean_R`: monodisperse population with the measured `Nv` and mean R.
- `Nv_plus_mean_R_plus_CV_lognormal`: lognormal population matching measured
  `Nv`, mean R, and population CV; 64-point Gauss-Hermite
  quadrature evaluates its scattering integral.
- `Sv_geometric_limit`: the established high-frequency geometric limit using
  `Sv` alone.
- `Sv_plus_M6`: the established equivalent population matching both moments.
- `full_PSD`: direct sum over every resolved particle; this is the reference.

The transport host is the frozen Yu 48 h non-particle baseline with `A_N=1.5`,
S11/S13 dislocation scattering off, and no refitted scale.  Seven transport
temperatures from 300 to 600 K are evaluated.  `fixed_6h_matrix` isolates PSD
evolution; `pf_time_varying_matrix` also propagates the PF far-field Ag value
through point-defect scattering.

## Fixed 6 h matrix results

| Descriptor | absolute kappa MAPE | maximum kappa error | delta-kappa signal NRMSE | max abs delta error (W/mK) | exact A/B/C delta ordering |
|---|---:|---:|---:|---:|---:|
| Nv_plus_mean_R | 0.287194% | 0.710164% | 349.712% | 0.0138937 | 0.312 |
| Nv_plus_mean_R_plus_CV_lognormal | 0.032739% | 0.124028% | 57.422% | 0.00280352 | 0.500 |
| Sv_geometric_limit | 7.133677% | 8.850038% | 889.846% | 0.0353669 | 0.312 |
| Sv_plus_M6 | 0.088457% | 0.176273% | 82.543% | 0.00289222 | 0.438 |
| full_PSD | 0.000000% | 0.000000% | 0.000% | 0 | 1.000 |

## PF time-varying matrix results

| Descriptor | absolute kappa MAPE | maximum kappa error | delta-kappa signal NRMSE | max abs delta error (W/mK) | exact A/B/C delta ordering |
|---|---:|---:|---:|---:|---:|
| Nv_plus_mean_R | 0.288952% | 0.715875% | 30.172% | 0.0138193 | 0.473 |
| Nv_plus_mean_R_plus_CV_lognormal | 0.033047% | 0.125780% | 4.977% | 0.00280357 | 0.625 |
| Sv_geometric_limit | 7.195025% | 8.914712% | 77.380% | 0.0354269 | 0.562 |
| Sv_plus_M6 | 0.089010% | 0.177729% | 7.114% | 0.00287439 | 1.000 |
| full_PSD | 0.000000% | 0.000000% | 0.000% | 0 | 1.000 |

## Interpretation

- Best compressed model by absolute-kappa MAPE: `Nv_plus_mean_R_plus_CV_lognormal`.
- Best compressed model for the 6--8 h evolution signal: `Nv_plus_mean_R_plus_CV_lognormal`.
- Absolute-kappa errors are diluted by the common host scattering background;
  the delta-kappa signal NRMSE is the stricter test of whether a descriptor
  preserves microstructure-evolution information.
- `Nv + mean R + CV` is not a unique PSD without a shape closure.  Its result
  here is conditional on the explicitly registered lognormal assumption.
- This 2 h, narrow-initial-PSD window can rank the candidates, but it cannot
  establish universal sufficiency for the much broader 6--48 h production
  ensemble.  The same comparison must be repeated on the completed Method-1
  A/B/C trajectories before freezing a final minimal descriptor.

```text
status=PASS_PF_ABC_6H8H_DESCRIPTOR_SUFFICIENCY_V1
snapshot_count=51
replicate_count=3
temperatures_K=300,350,400,450,500,550,600
full_PSD_reference=true
dislocation_mode=DISLOCATION_OFF
absolute_experimental_kappa_claim=false
running_PF_jobs_modified=false
```
