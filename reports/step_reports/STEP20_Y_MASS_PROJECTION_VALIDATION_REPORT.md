# Step 20 Y Mass Projection Validation Report

## Purpose

This round formalizes the Step 19 `pre_Y_update` scalar projection as a **conservative correction candidate** for post-conversion `Y_update_k0_drift`.

The projection remains **optional** and **default-off**.

## Scope Guard

This round did **not** change the default behavior of:

- stochastic trigger
- eligible detection
- random number generation
- GP-to-beta conversion operator
- local compensation logic
- `storage_exact`
- eta PDE
- phi PDE
- transport solver
- `two_phase`
- default clipping path

## Modified Files

- `/Users/heng/Documents/GitHub/CUDA_STO_PF/pf_params.h`
- `/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu`
- `/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.h`
- `/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu`
- `/Users/heng/Documents/GitHub/CUDA_STO_PF/run_step20_y_mass_projection_validation.sh`

## Final Parameter Set

- `y_update_mass_projection_enabled`
- `y_update_mass_projection_report_enabled`
- `y_update_mass_projection_max_iter`
- `y_update_mass_projection_tol`
- `y_update_mass_projection_target_mode = pre_Y_update | post_conversion_baseline`
- Existing audit parameters retained:
  - `y_update_k0_audit_enabled`
  - `y_update_k0_audit_steps`
  - `y_update_k0_audit_prefix`

Recommended mode:

- `y_update_mass_projection_enabled = 1`
- `y_update_mass_projection_target_mode = pre_Y_update`

## Algorithm

After each Y update, when enabled:

`Y_i <- Y_i + lambda`

Choose `lambda` so that:

`sum_xBtot(storage_exact(phi, eta, xB_alpha(sigmoid(Y+lambda)))) = target_sum_xBtot`

Target modes:

- `pre_Y_update`
  - restore the substep-preceding conserved mass
  - recommended production candidate
- `post_conversion_baseline`
  - restore the accepted conversion-event baseline mass
  - retained as debug/test mode

Solver notes:

- bracket starts at `[-50, 50]`
- expands to `[-100, 100]` if needed
- uses mass computed through the same `storage_exact` relation as the main code
- does not project by clipping `xB`

## Output Directories

- Case 0:
  `/tmp/step20_y_projection_validation/runs/default_off_regression/Results/chel_T400_cuda_32x32x32_dt1e-05_steps50_xB0.030/Step20_default_off_50`
- Case 1:
  `/tmp/step20_y_projection_validation/runs/projection_preY_50/Results/chel_T400_cuda_32x32x32_dt1e-05_steps50_xB0.030/Step20_projection_preY_50`
- Case 2:
  `/tmp/step20_y_projection_validation/runs/projection_preY_1/Results/chel_T400_cuda_32x32x32_dt1e-05_steps1_xB0.030/Step20_projection_preY_1`
- Case 3:
  `/tmp/step20_y_projection_validation/runs/projection_baseline_50/Results/chel_T400_cuda_32x32x32_dt1e-05_steps50_xB0.030/Step20_projection_baseline_50`
- Case 4:
  `/tmp/step20_y_projection_validation/runs/no_event_projection_on/Results/chel_T400_cuda_32x32x32_dt1e-05_steps50_xB0.030/Step20_no_event_projection_on`

## Case 0: Default-Off Regression

MD5 vs Step 18 / Step 19 passive path:

- `phi_50.vtk`: match
- `xB_50.vtk`: match
- `xBtot_gp_50.vtk`: match
- `eta_50.vtk`: match

Drift is unchanged:

- `total_relative_drift = 6.115772602651491e-03`

Conclusion:

- default-off path is bitwise preserved

## Case 1: Projection `pre_Y_update`, 50 Steps

Result:

- `total_relative_drift = -1.8946299221368732e-09`
- `total_absolute_drift = -5.683920695220834e-11`

Reduction factor vs default-off:

- `6.115772602651491e-03 / 1.8946299221368732e-09 ≈ 3.23e+06`

Projection statistics:

- rows: `50`
- converged rows: `48`
- max projection residual: `6.603485155665112e-05`
- lambda range:
  - `min = -6.254483014345169e-03`
  - `max = 0`

State health:

- `min_xB_after_projection = 9.93765035612458382e-09`
- `max_xB_after_projection = 9.58155693893267962e-02`
- no projection-induced clipping
- storage residual from audit:
  - `max storage_residual_sum_after = 0`
  - `max storage_residual_max_abs_after = 0`

Note on clip counts:

- `total_clip_count_xB = 34`, same as the old forced-event path
- projection did not add a new clipping problem; it suppressed the drift without increasing clipping counts

## Case 2: Projection `pre_Y_update`, 1 Step

The first raw Y-update jump is still visible:

- `delta_sum_xBtot_Y_update = +5.93997273402078463`

After projection:

- `delta_sum_xBtot_Y_update = -6.60348515566511196e-05`
- `total_relative_drift = -4.3739153153888924e-08`

Interpretation:

- the first-step jump is effectively removed immediately
- residual is small and bounded
- no NaN
- no projection-induced clipping
- storage residual remains zero

## Case 3: Projection `post_conversion_baseline`, 50 Steps

Result:

- `total_relative_drift = -1.8967329737871882e-09`
- `total_absolute_drift = -5.690229884502962e-11`

This mode is also effective, but operationally less clean:

- converged rows: `0`
- repeatedly hits `max_iter_reached`
- oscillates around the baseline with small residuals

State health remains good:

- no NaN
- no additional clipping
- storage residual remains zero

Comparison to `pre_Y_update`:

- both suppress drift to `~1e-9`
- `pre_Y_update` is preferred because it behaves more naturally as a substep-local conservation correction and reaches tolerance in most later steps

## Case 4: No-Event Baseline With Projection Enabled

Result:

- `total_relative_drift = 0`
- `total_absolute_drift = 0`
- `total_clip_count_xB = 0`
- `total_clip_count_Y = 0`
- `max_abs_mass_closure_resid = 0`

Event status:

- `gp_to_beta_events.csv` contains candidate checks
- `event_accepted = 0` for all 50 rows
- `y_update_mass_projection.csv` contains header only

Interpretation:

- when there is no accepted conversion event, projection does not activate and does not perturb the no-event dynamics

## Optional Larger Sanity Test

Not run in this round.

Reason:

- the current objective was to formalize and validate the correction candidate on the already diagnosed `32^3` forced-event regression case first

## Conclusion

`pre_Y_update` scalar projection is now a credible **formal conservative correction candidate**:

- default-off path is unchanged
- 50-step forced-event drift drops from `6.1157726e-03` to `1.8946299e-09`
- no NaN
- no new clipping issue
- storage residual remains zero
- no-event baseline remains healthy

## Recommendation

Yes: `pre_Y_update` projection should be kept as the **formal conservative correction candidate** for the post-conversion Y-update path.

Current recommendation:

- keep it **default-off**
- use it as the preferred repair mode for GP->beta post-conversion dynamics validation
- validate next on:
  - larger grids
  - scheduled insertion / production-like event sequences
  - more realistic multi-event runs

## Limitation

- projection is still optional and not yet the production default
- the internal decomposition of `RHS_k0_total` is still incomplete
- the correction is validated strongly on the diagnosed `32^3` regression, but not yet on larger production-like workloads
