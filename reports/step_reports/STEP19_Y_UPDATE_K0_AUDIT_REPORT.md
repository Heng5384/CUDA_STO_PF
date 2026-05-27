# Step 19 Y-Update k=0 Audit Report

## Scope

This round did **not** modify the default physics path for:

- stochastic trigger
- GP-to-beta conversion operator
- `storage_exact`
- eta PDE
- phi PDE
- transport solver
- `two_phase`

All new functionality is default-off diagnostic or test-only projection logic.

## New Parameters

- `y_update_k0_audit_enabled`
- `y_update_k0_audit_steps`
- `y_update_k0_audit_prefix`
- `y_update_mass_projection_enabled`
- `y_update_mass_projection_max_iter`
- `y_update_mass_projection_tol`
- `y_update_mass_projection_target_mode = pre_Y_update | post_conversion_baseline`

## New CSV Outputs

- `y_update_k0_audit.csv`
- `y_update_mass_projection.csv`

## Case Output Directories

- Case A:
  `/tmp/step19_y_k0_audit/runs/audit_1step/Results/chel_T400_cuda_32x32x32_dt1e-05_steps50_xB0.030/Step19_audit_1`
- Case B:
  `/tmp/step19_y_k0_audit/runs/audit_50steps/Results/chel_T400_cuda_32x32x32_dt1e-05_steps50_xB0.030/Step19_audit_50`
- Case C:
  `/tmp/step19_y_k0_audit/runs/projection_preY_50steps/Results/chel_T400_cuda_32x32x32_dt1e-05_steps50_xB0.030/Step19_projection_preY_50`
- Case D:
  `/tmp/step19_y_k0_audit/runs/projection_baseline_50steps/Results/chel_T400_cuda_32x32x32_dt1e-05_steps50_xB0.030/Step19_projection_baseline_50`

## Case A: First Post-Conversion Y Update

The first post-conversion `after_Y_update` row now reproduces the Step 18 jump exactly:

| quantity | value |
|---|---:|
| `sum_xBtot_before_Y` | `9.83045372236304956e+02` |
| `sum_xBtot_after_Y` | `9.88985344970325741e+02` |
| `delta_sum_xBtot_Y_update` | `+5.93997273402078463e+00` |
| `Y_k0_before` | `-1.17251970394719712e+05` |
| `Y_k0_after` | `-1.14444524477181432e+05` |
| `delta_Y_k0` | `+2.80744591753827990e+03` |
| `RHS_k0_total` | `+2.80744591753827989e+08` |
| `storage_residual_sum_before/after` | `0 / 0` |
| `storage_residual_max_abs_before/after` | `0 / 0` |

Current decomposition status:

- `RHS_k0_total` is available and matches the observed drift source.
- `RHS_k0_linear`, `RHS_k0_nonlinear`, `RHS_k0_stabilization_add`, `RHS_k0_stabilization_subtract`, `RHS_k0_source`, `RHS_k0_transport` are still `NaN` in this first implementation.
- `RHS_k0_unknown` is set equal to `RHS_k0_total`, so the unresolved k=0 contribution is still fully captured.

## Case B: Passive Audit Validation

Passive audit does not change the accepted Step 18 `P=1` path.

MD5 vs Step 18 `P1_50`:

| file | result |
|---|---|
| `phi_50.vtk` | match |
| `xB_50.vtk` | match |
| `xBtot_gp_50.vtk` | match |
| `eta_50.vtk` | match |

50-step drift is reproduced exactly:

- `total_relative_drift = 6.115772602651491e-03`
- `total_absolute_drift = 1.8347417644639166e-04`
- `suspected_primary_source = Y_update_k0_drift`

## Case C: Projection Enabled, `pre_Y_update`

Result:

- `total_relative_drift = -1.8946299221368732e-09`
- `total_absolute_drift = -5.683920695220834e-11`

Reduction factor vs passive 50-step drift:

- `6.115772602651491e-03 / 1.8946299221368732e-09 ≈ 3.23e+06`

Projection behavior:

- step 1 projection target is now valid and equals `sum_xBtot_before_Y`
- step 1 residual after projection:
  `-6.60348515566511196e-05`
- most later steps are already within tolerance
- 48/50 rows report `converged=1`
- no clipping introduced
- `min_xB_after_projection = 9.93765035612458382e-09`
- `max_xB_after_projection = 9.58155693893267962e-02`

## Case D: Projection Enabled, `post_conversion_baseline`

Result:

- `total_relative_drift = -1.8967329737871882e-09`
- `total_absolute_drift = -5.690229884502962e-11`

Reduction factor vs passive 50-step drift:

- `6.115772602651491e-03 / 1.8967329737871882e-09 ≈ 3.22e+06`

Projection behavior:

- step 1 residual after projection:
  `-6.60348515566511196e-05`
- later rows oscillate around the conversion baseline with residuals on the order of `2.5e-05` to `6.3e-05`
- `converged=0` in this implementation because the current bracket/bisection hits `max_iter_reached` before the strict `1e-12` target
- despite that, global drift is still suppressed to `O(1e-9)`
- no clipping introduced
- `min_xB_after_projection = 9.93765035612458382e-09`
- `max_xB_after_projection = 9.58155700464801419e-02`

## Interpretation

### Is Y k=0 update the direct source?

Yes.

The first post-conversion Y update contributes:

- `delta_sum_xBtot_Y_update = +5.93997273402078463`

This is the same quantity previously identified in Step 18 as the dominant first-step jump, and it appears directly in the new Step 19 `y_update_k0_audit.csv`.

### Does scalar conservative projection work?

Yes, as a repair candidate.

Both projection modes reduce the 50-step relative drift from:

- `6.115772602651491e-03`

to:

- `1.8946299221368732e-09` (`pre_Y_update`)
- `1.8967329737871882e-09` (`post_conversion_baseline`)

with:

- no clipping
- no NaN/Inf
- no storage residual anomaly

### Which projection mode looks better right now?

`pre_Y_update` is cleaner operationally in the current implementation:

- it now has a valid first-step target
- most later steps are already within tolerance
- it reaches `converged=1` on 48/50 rows

`post_conversion_baseline` is also effective for drift suppression, but the current solver settings leave it bouncing around the baseline with small residuals and `max_iter_reached`.

## Next Recommendation

The evidence now supports treating scalar conservative projection as a serious repair candidate for the post-conversion Y-update path.

Recommended next step:

1. keep the current projection path default-off
2. formalize `pre_Y_update` projection as the primary conservative correction candidate
3. if we want a default-on physics fix later, first split `RHS_k0_total` into source/stabilization/transport pieces to understand why the first post-conversion k=0 update is so large

