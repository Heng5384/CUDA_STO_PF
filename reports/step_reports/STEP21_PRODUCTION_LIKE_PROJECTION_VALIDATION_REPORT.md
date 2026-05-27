# Step 21 Production-Like Projection Validation Report

## Goal

Validate whether the Step 20 `pre_Y_update` conservative scalar projection remains useful in more production-like `gp_zone` event simulations, while keeping:

- `y_update_mass_projection_enabled = 0` by default
- `storage_exact` unchanged
- Step 13–20 default paths unchanged

## Scope

This round did not modify:

- stochastic trigger logic
- conversion operator
- `storage_exact` main formula
- eta PDE
- phi PDE
- transport solver
- `two_phase`

The only non-default path exercised was:

- `y_update_mass_projection_enabled = 1`
- `y_update_mass_projection_target_mode = pre_Y_update`

## Modified Files

- `/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu`
- `/Users/heng/Documents/GitHub/CUDA_STO_PF/run_step21_production_projection_validation.sh`

## Output Root

- `/tmp/step21_projection_validation/runs`

## Test Matrix

### 1. Larger-grid single-event sanity

- `64^3`
- one accepted `GP -> beta` conversion
- 50 and 200 steps
- off/on comparison

Runs:

- `larger64_single_event_off_50`
- `larger64_single_event_on_50`
- `larger64_single_event_off_200`
- `larger64_single_event_on_200`

### 2. Multi-event production-like surrogate

Current code has no explicit scheduled 3-event insertion path.

To approximate the requested “3 separated events at separated times and locations” without changing the main solver, I used a reproducible fixed-seed surrogate based on existing event machinery:

- `gp_init_mode = none`
- `gp_nuc_enabled = 1`
- accepted nucleation events at separated check times
- `gp_to_beta_enabled = 1`

Two variants were checked:

- an extreme event-heavy stress case with accepted events at essentially every check
- a 3-event surrogate (`gp_nuc_check_interval = 60`) giving exactly 3 accepted `GP -> beta` conversions at separated times

### 3. Stochastic GP nucleation only

- `gp_nuc_enabled = 1`
- `gp_to_beta_enabled = 0`
- compare projection off/on

Runs:

- `stochastic_nuc_off_200`
- `stochastic_nuc_on_200`

## Results

### A. Larger-grid single-event sanity

#### 64^3, 50 steps

- off:
  - `total_relative_drift = 7.595936176147714e-04`
- on:
  - `total_relative_drift = 5.19641283892081e-08`

Drift reduction factor:

- about `1.46e+04`

State health:

- no NaN/Inf
- no new clipping
- `xB_clip_count_high = 0`
- `gp_closure_error ~ 4.46e-21`
- `storage residual = 0`

#### 64^3, 200 steps

- off:
  - `total_relative_drift = 7.595936176147714e-04`
- on:
  - `total_relative_drift = 5.196412862050442e-08`

Drift reduction factor:

- about `1.46e+04`

State health:

- no NaN/Inf
- no new clipping
- `xB_clip_count_high = 0`
- `gp_closure_error ~ 3.75e-20`
- `storage residual = 0`

Conclusion:

- `pre_Y_update` projection remains effective on the larger-grid single-event case

### B. Stochastic GP nucleation only

#### 32^3, 200 steps, `gp_nuc_enabled=1`, `gp_to_beta_enabled=0`

off:

- `total_relative_drift = -2.7755575615628914e-15`
- `gp_nucleation_events.csv rows = 200`
- `eta_integral = 1.2777231675e-01`
- `xB_clip_count_high = 0`

on:

- `total_relative_drift = -2.7755575615628914e-15`
- `gp_nucleation_events.csv rows = 200`
- `eta_integral = 1.2777231675e-01`
- `xB_clip_count_high = 0`

Conclusion:

- with no accepted `GP -> beta` conversion, projection does not perturb the nucleation-only dynamics
- no NaN/Inf
- no clipping
- no storage residual issue

### C. Event-heavy accepted-conversion stress case

#### 32^3, 200 steps, accepted events every check

off:

- `total_relative_drift = 9.564021846402063`
- `gp_to_beta accepted events = 200`
- `gp_nucleation_events.csv rows = 200`
- `total_clip_count_xB = 281889`
- `gp_closure_error = 5.8605764930e-04`

on:

- `total_relative_drift = -1.499811896630704e-03`
- `gp_to_beta accepted events = 200`
- `gp_nucleation_events.csv rows = 187`
- `total_clip_count_xB = 3111457`
- `gp_closure_error = 5.2715187862e-05`

Interpretation:

- projection reduces drift magnitude strongly compared with the raw runaway case
- but it also drives **much larger clipping**
- this is not acceptable as a production-ready validation pass

### D. Three-event separated surrogate

Using:

- `gp_nuc_check_interval = 60`
- `gp_nuc_J0 = 1e38`
- `temperature_C = 27`

Accepted `GP -> beta` events occurred at:

- step 60
- step 120
- step 180

off:

- `total_relative_drift = 8.350557821307913`
- `accepted conversions = 3`
- `total_clip_count_xB = 177852`

on:

- `total_relative_drift = 7.584659623642447`
- `accepted conversions = 3`
- `total_clip_count_xB = 230607`

Interpretation:

- this is closer to the requested multi-event scenario than the fully event-heavy stress case
- but `pre_Y_update` projection still does **not** produce the needed drift suppression
- clipping also gets worse rather than better

## Overall Assessment

### What passed

1. **Larger-grid single-event sanity**
   - yes
   - drift reduced by more than `1e4`
   - no NaN
   - no significant new clipping
   - storage residual remains zero

2. **Stochastic nucleation only**
   - yes
   - projection-on path is effectively inert when no conversion is accepted
   - no drift penalty
   - no NaN
   - no clipping

### What did not pass

1. **Accepted multi-event production-like runs**
   - no
   - both the 3-event surrogate and the event-heavy stress case still show large drift or clipping pathology
   - projection is not yet robust enough for event-heavy `gp_zone` production-like runs

## Conclusion

`pre_Y_update` projection is validated as an **optional conservative correction candidate** for:

- single accepted conversion runs
- larger-grid single-event sanity checks

It is **not yet validated** for:

- repeated accepted `GP -> beta` conversion runs
- event-heavy production-like scenarios

So the correct Step 21 conclusion is:

- keep `y_update_mass_projection_enabled = 0` by default
- keep `pre_Y_update` as the recommended optional mode for targeted post-conversion drift suppression
- do **not** yet promote it as a generally safe correction for event-heavy production runs

## Recommended Next Step

Focus the next round on why repeated accepted conversions still accumulate clipping under projection. Likely areas:

1. projection activation horizon vs repeated event resets
2. interaction between repeated local compensation and subsequent Y-shift correction
3. repeated-event clipping path, which is now the dominant failure mode in production-like runs

