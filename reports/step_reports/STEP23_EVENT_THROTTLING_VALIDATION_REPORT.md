# Step 23 Event Throttling Validation

## Scope

This step adds **event throttling / spacing control** before accepted `GP -> beta` conversion events.

What was intentionally left unchanged:

- `storage_exact` formulation
- transport PDE
- elastic solver
- Step 13 stability chain
- Step 14 elastic chain
- Step 15 conversion operator
- stochastic GP nucleation logic
- `two_phase` path
- Step 22 feasibility gate logic

The only new logic is a **host-side pre-event throttling filter**.

## Files changed

- `/Users/heng/Documents/GitHub/CUDA_STO_PF/pf_params.h`
- `/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu`
- `/Users/heng/Documents/GitHub/CUDA_STO_PF/run_step23_event_throttling_validation.sh`

## New parameters

- `gp_to_beta_event_cooldown_steps`
- `gp_to_beta_min_event_spacing`
- `gp_to_beta_event_exclusion_radius`
- `gp_to_beta_max_events_global`
- `gp_to_beta_max_events_per_window`
- `gp_to_beta_event_window_steps`

Defaults keep Step 22 behavior unchanged:

- `cooldown = 0`
- `spacing = 0`
- `exclusion_radius = 0`
- `max_events_global = 1e9`
- `max_events_per_window = 1e9`
- `window_steps = 0`

## Event log additions

`gp_to_beta_events.csv` now includes:

- `throttling_checked`
- `rejected_cooldown`
- `rejected_spacing`
- `rejected_window_limit`
- `rejected_global_limit`
- `nearest_event_distance`
- `steps_since_nearest_event`
- `accepted_events_in_window`
- `accepted_events_total`

## Runtime counters

Added:

- `gp_to_beta_events_rejected_cooldown`
- `gp_to_beta_events_rejected_spacing`
- `gp_to_beta_events_rejected_window`
- `gp_to_beta_events_rejected_global`
- `gp_to_beta_events_accepted_total`

## Validation runs

Root:

- `/tmp/step23_event_throttling_validation/runs`

Cases:

1. `case0_no_throttle_regression`
2. `case1_single_event_throttle`
3. `case2_three_events_throttle`
4. `case3_event_heavy_throttle`
5. `case4_stochastic_nuc_only`

## Results

### Case 0: no-throttle regression

Purpose:

- confirm default throttling-off behavior is unchanged

Result:

- bitwise identical to Step 22 `event_heavy_gate_on_200`
- MD5 matched for:
  - `phi_200.vtk`
  - `xB_200.vtk`
  - `xBtot_gp_200.vtk`
  - `eta_200.vtk`

Mass/clipping:

- `total_relative_drift = 9.704025619799014`
- `total_clip_count_xB = 294259`
- accepted events = `200`
- rejected cooldown/spacing/window/global = `0/0/0/0`

Conclusion:

- throttling defaults are passive

### Case 1: single-event sanity

Configuration:

- throttling enabled
- only one conversion event exists

Result:

- bitwise identical to Step 22 `single64_gate_on_200`
- accepted events = `1`
- rejected cooldown/spacing/window/global = `0/0/0/0`
- scaled events = `1`

Mass/clipping:

- `total_relative_drift = 6.805459850039106e-04`
- `total_clip_count_xB = 34`
- accepted-event `max_abs(mass_error_comp) = 1.6479873022e-17`

Conclusion:

- throttling does not perturb the healthy single-event path

### Case 2: three separated events

Configuration:

- moderate throttling
- events are already well separated in time/space

Result:

- accepted events = `3`
- rejected cooldown/spacing/window/global = `0/0/0/0`
- scaled events = `1`

Mass/clipping:

- `total_relative_drift = 7.392016893910239`
- `total_clip_count_xB = 177339`
- accepted-event `max_abs(mass_error_comp) = 6.9388939039e-17`

Conclusion:

- all 3 events still survive
- throttling does not make this surrogate worse
- behavior matches Step 22 gate-on three-event case

### Case 3: event-heavy stress

Configuration:

- `cooldown = 20`
- `min_event_spacing = 0.4`
- `event_exclusion_radius = 0.4`
- `max_events_global = 20`
- `max_events_per_window = 1`
- `window_steps = 20`

Result:

- rows logged = `200`
- accepted events = `10`
- rejected cooldown = `189`
- rejected spacing = `9`
- rejected window = `189`
- rejected global = `0`
- infeasible rejects = `0`
- scaled events = `1`

Mass/clipping:

- `total_relative_drift = 8.70255420597658`
- `total_clip_count_xB = 214902`
- accepted-event `max_abs(mass_error_comp) = 3.0531133177e-16`
- `max_abs_delta_gp_to_beta_event = 2.0916069987109154e-09`

Comparison against Step 22 event-heavy gate-on:

- Step 22 accepted events: `200`
- Step 23 accepted events: `10`
- Step 22 `xB` clips: `294259`
- Step 23 `xB` clips: `214902`
- clip reduction: about `26.9%`
- Step 22 drift: `9.704025619799014`
- Step 23 drift: `8.70255420597658`

Conclusion:

- throttling is doing the intended physical job
- it suppresses unrealistic conversion density
- accepted events stay locally conservative
- clipping and drift are reduced materially
- but not eliminated

### Case 4: stochastic GP nucleation only

Configuration:

- `gp_to_beta_enabled = 0`

Result:

- `total_relative_drift = -2.7755575615628914e-15`
- `total_clip_count_xB = 0`
- no accepted conversion events

Conclusion:

- stochastic nucleation-only baseline remains unchanged

## Overall conclusion

Step 23 passes its intended goal:

- default-off path is unchanged
- rejection path is explicitly logged
- event-heavy repeated conversions are throttled in a physically meaningful way
- accepted-event compensation remains near numerical tolerance
- no new solver-path bug was introduced

What this step does **not** claim:

- it does not fully eliminate event-heavy drift
- it does not replace the feasibility gate
- it does not make global projection unnecessary in every extreme forced-event regime

What it does establish:

> we are no longer using solver-side rescue to cope with obviously over-dense conversion bursts; we now have an explicit event-statistics control layer that suppresses those bursts before conversion.

## Recommendation

Keep:

- projection default `off`
- feasibility gate `on`
- throttling parameters available and default-passive

Recommended next direction:

- calibrate throttling parameters for production-like GP-zone runs
- especially `cooldown`, `exclusion_radius`, and `max_events_per_window`
- then re-evaluate whether optional projection is still needed under those more realistic event statistics
