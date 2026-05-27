# Step 18 Post-Conversion Y-Update Audit

## Scope

This round only added passive diagnostics for post-conversion PDE evolution after an accepted GP->beta event.

Unchanged on purpose:

- stochastic trigger
- GP->beta conversion operator
- `storage_exact`
- `eta` PDE
- transport solver
- `two_phase`

## Added parameters

- `post_conversion_y_update_audit_enabled`
- `post_conversion_y_update_audit_steps`
- `post_conversion_y_update_audit_prefix`

## Added outputs

- `post_conversion_y_update_audit.csv`

## Cases

- Case A: `P=1`, audit 1 step
  - `/tmp/step18_postconv_audit/runs/p1_1/Results/chel_T400_cuda_32x32x32_dt1e-05_steps50_xB0.030/Step18_P1_1`
- Case B: `P=1`, audit 5 steps
  - `/tmp/step18_postconv_audit/runs/p1_5/Results/chel_T400_cuda_32x32x32_dt1e-05_steps50_xB0.030/Step18_P1_5`
- Case C: `P=1`, audit 50 steps
  - `/tmp/step18_postconv_audit/runs/p1_50/Results/chel_T400_cuda_32x32x32_dt1e-05_steps50_xB0.030/Step18_P1_50`

## Passive check

Step 18 audit is passive. For Case C, the final field MD5 values match the prior Step 17 `P=1` run exactly:

- `phi_50.vtk`
- `xB_50.vtk`
- `xBtot_gp_50.vtk`
- `eta_50.vtk`

The run still reproduces the known 50-step drift:

- `total_relative_drift = 6.115772602651491e-03`
- primary source in the existing mass summary remains `Y_update_k0_drift`

## Stage table

### Case A: first audited step

| step | stage | `delta_sum_xBtot_from_previous_stage` | `delta_sum_xBtot_from_post_conversion_baseline` | `Y_update_k0_drift` | `clipped_mass_loss` | `num_clipped_high` |
|---|---|---:|---:|---:|---:|---:|
| 1 | `before_step` | `-2.303728047082e-05` | `-2.303728047082e-05` | `0` | `0` | `0` |
| 1 | `after_phi_update` | `+2.303728047082e-05` | `0` | `0` | `0` | `0` |
| 1 | `after_Y_update` | `+5.939972734021e+00` | `+5.939972734021e+00` | `+5.939972734021e+00` | `0` | `0` |
| 1 | `after_Y_to_xB` | `0` | `+5.939972734021e+00` | `+5.939972734021e+00` | `0` | `0` |
| 1 | `after_clipping` | `+2.273736754432e-13` | `+5.939972734021e+00` | `+5.939972734021e+00` | `+2.273736754432e-13` | `33` |
| 1 | `end_of_step` | `0` | `+5.939972734021e+00` | `+5.939972734021e+00` | `+2.273736754432e-13` | `33` |

### Case B: first 5 audited steps

Observed pattern:

- step 1:
  - dominant jump appears at `after_Y_update`: `+5.939972734018e+00`
- step 2:
  - `after_phi_update`: `-5.686741832426e-02`
  - `after_Y_update`: `+1.289534608180e-01`
- step 3:
  - `after_phi_update`: `-2.407454489685e-03`
  - `after_Y_update`: `+2.407454488775e-03`
- step 4:
  - `after_phi_update`: `-8.698831603624e-04`
  - `after_Y_update`: `+8.698831703668e-04`
- step 5:
  - `after_phi_update`: `-4.088030702860e-04`
  - `after_Y_update`: `+4.088030732419e-04`

In every audited step:

- `after_Y_to_xB` adds `0`
- `after_clipping` adds only roundoff-scale mass change, around `1e-12`
- `storage_residual_sum = 0`
- `storage_residual_max_abs = 0`

### Case C: 50 audited steps

Key numbers:

- first nontrivial PDE drift stage:
  - step `1`, stage `after_Y_update`, delta `+5.939972734013509`
- maximum `Y_update_k0_drift`:
  - step `1`, stage `after_Y_update`, magnitude `5.939972734013509`
- maximum `|clipped_mass_loss|`:
  - step `18`, stage `after_clipping`, magnitude `1.0345502232667059e-11`
- final `delta_sum_xBtot_from_post_conversion_baseline` at step 50:
  - `+6.012058776514891`

The first-step `after_Y_update` jump already accounts for about `98.8%` of the final post-conversion baseline shift:

- `5.9399727340207846 / 6.0120587765148912 = 0.9880`

## Findings

1. Drift does **not** start in `after_phi_update`.
   - The phi substep produces only a small offset.
   - Step 1: `after_phi_update` is exactly at the post-conversion baseline.
   - Later steps show small `after_phi_update` corrections, but they are two to four orders of magnitude smaller than the first `after_Y_update` jump.

2. Drift starts in `after_Y_update`.
   - The first large mass jump appears at:
     - step `1`
     - stage `after_Y_update`
     - `delta_sum_xBtot_from_previous_stage = +5.939972734013509`

3. `Y_update_k0_drift` matches the mass jump.
   - In the same step/stage:
     - `Y_update_k0_drift = +5.939972734013509`
   - This is the same order as the total 50-step drift.

4. `after_Y_to_xB` is not adding extra drift.
   - Its stage delta is `0` in all audited steps.

5. Clipping is not the primary source of the drift.
   - Clipping counts appear immediately after conversion, but the associated mass loss stays at roundoff scale.
   - Largest observed `|clipped_mass_loss|` over 50 audited steps is only `1.03e-11`.

6. Storage inconsistency is not the source.
   - `storage_residual_sum = 0`
   - `storage_residual_max_abs = 0`
   - throughout the audited stages.

## Conclusion

The post-conversion drift starts at the **first `after_Y_update` substep** after the accepted GP->beta event.

Current evidence is consistent with:

- conversion instant: basically conservative
- clipping: not the dominant source
- storage residual: not the source
- main source: `Y_update_k0_drift`

## Next step

Focus only on the Y-update chain immediately after conversion, especially the `k=0` handling in the first post-conversion step.
