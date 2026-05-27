# Step 22 Feasibility Gate Validation

## Scope

This round added a **pre-event local feasibility gate** for repeated `GP -> beta` conversions in `model_mode=gp_zone`.

What stayed unchanged:

- `storage_exact` main formula
- Step 13 transport
- Step 14 elastic
- Step 15 conversion operator itself
- `two_phase` path
- global `pre_Y_update` mass projection default (`off`)

The new logic only checks whether the proposed conversion can be compensated by the local shell without forcing `xB_alpha` outside `[xB_min, xB_max]`.

## Files changed

- `/Users/heng/Documents/GitHub/CUDA_STO_PF/pf_params.h`
- `/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu`
- `/Users/heng/Documents/GitHub/CUDA_STO_PF/run_step22_feasibility_gate_validation.sh`

## New parameters

- `gp_to_beta_feasibility_gate_enabled = 1`
- `gp_to_beta_min_shell_capacity_factor = 1.05`
- `gp_to_beta_reject_if_infeasible = 1`
- `gp_to_beta_allow_seed_amplitude_scaling = 1`
- `gp_to_beta_min_seed_amplitude = 0.05`
- `gp_to_beta_event_cooldown_steps = 0`
- `gp_to_beta_min_event_spacing = 0`

## Event log additions

`gp_to_beta_events.csv` now includes:

- `feasibility_checked`
- `mass_error_raw_predicted`
- `shell_capacity_add`
- `shell_capacity_remove`
- `capacity_ratio`
- `event_rejected_infeasible`
- `seed_amplitude_original`
- `seed_amplitude_scaled`
- `seed_scaled_due_to_capacity`
- `cooldown_or_spacing_rejected`
- `xB_shell_min_before_event`
- `xB_shell_max_before_event`

## Validation runs

Run root:

- `/tmp/step22_feasibility_gate_validation/runs`

Cases:

1. `single64_gate_off_200`
2. `single64_gate_on_200`
3. `three_event_gate_off_200`
4. `three_event_gate_on_200`
5. `event_heavy_gate_off_200`
6. `event_heavy_gate_on_200`
7. `stochastic_nuc_only_gate_on_200`

Additional rejection smoke test:

- `/tmp/step22_reject_smoke`

## Key results

### 1. Single-event sanity

This case turned out to be **locally infeasible without the gate**.

`single64_gate_off_200`:

- accepted events: `1`
- capacity ratio min: `0.38273731423`
- `mass_error_comp = 1.0756965573e-02`
- `xB_clip_count_event = 224`
- 200-step `total_relative_drift = 2.0642946526727886e-03`
- total `xB` clips: `262`

`single64_gate_on_200`:

- accepted events: `1`
- scaled events: `1`
- capacity ratio min: `1.05`
- `seed_amplitude_scaled = 5.9912762346e-01`
- `mass_error_comp = -1.6479873022e-17`
- `xB_clip_count_event = 0`
- 200-step `total_relative_drift = 6.805459850039106e-04`
- total `xB` clips: `34`

Interpretation:

- the gate did **not** leave this single-event case unchanged, because the original full seed was not locally feasible
- instead it scaled the seed to the minimum safe amplitude that satisfied the shell capacity margin
- this materially improved compensation and reduced clipping

### 2. Three separated surrogate events

`three_event_gate_off_200`:

- accepted: `3`
- scaled: `0`
- rejected capacity: `0`
- min capacity ratio: `0.65341565788`
- `total_relative_drift = 8.177795055841557`
- total `xB` clips: `177443`
- `max_abs_delta_gp_to_beta_event = 1.8055062585277393e-04`

`three_event_gate_on_200`:

- accepted: `3`
- scaled: `1`
- rejected capacity: `0`
- min capacity ratio: `1.05`
- `total_relative_drift = 7.392016893910239`
- total `xB` clips: `177339`
- `max_abs_delta_gp_to_beta_event = 2.177000246739169e-09`

Interpretation:

- the gate kept accepted conversions locally conservative
- drift improved modestly
- clipping improved only slightly in this surrogate
- the major gain here is that event-level compensation error dropped from `~1.8e-4` to `~2.2e-9`

### 3. Event-heavy stress

`event_heavy_gate_off_200`:

- accepted: `200`
- scaled: `0`
- rejected capacity: `0`
- min capacity ratio: `0.62289963089`
- `total_relative_drift = 1.0050773895525886e+01`
- total `xB` clips: `356651`
- `max_abs_delta_gp_to_beta_event = 1.966884765254029e-04`

`event_heavy_gate_on_200`:

- accepted: `200`
- scaled: `8`
- rejected capacity: `0`
- min capacity ratio: `1.05`
- `total_relative_drift = 9.704025619799014`
- total `xB` clips: `294259`
- `max_abs_delta_gp_to_beta_event = 3.098438727811015e-09`

Interpretation:

- gate-on reduced total `xB` clipping by about `17.5%`
- gate-on also restored accepted-event compensation to numerical tolerance
- but this did **not** fully cure the broader repeated-event instability

### 4. Stochastic GP nucleation only

`stochastic_nuc_only_gate_on_200`:

- `gp_to_beta_enabled = 0`
- `total_relative_drift = -2.7755575615628914e-15`
- `xB` clips: `0`
- no NaN/Inf

Interpretation:

- the new gate does not disturb the stochastic GP nucleation-only path

### 5. Explicit rejection-path smoke test

Ad hoc run with:

- `gp_to_beta_feasibility_gate_enabled = 1`
- `gp_to_beta_allow_seed_amplitude_scaling = 0`
- `gp_to_beta_reject_if_infeasible = 1`

Result:

- `gp_to_beta_events_accepted : 0`
- `gp_to_beta_events_rejected_capacity : 1`
- `total_relative_drift = 0`

Interpretation:

- infeasible events are rejected cleanly rather than silently ignored

## Overall conclusion

Step 22 is successful as a **local feasibility protection layer**:

- accepted events are now checked against shell capacity before conversion
- infeasible events can be scaled or rejected
- accepted events remain mass-conserving to numerical tolerance
- repeated-event clipping is reduced
- the nucleation-only path remains unchanged

But Step 22 is **not** a complete cure for event-heavy repeated-conversion dynamics:

- local infeasibility is no longer hidden
- event-level compensation is much healthier
- however, large repeated-event runs still accumulate substantial global drift and clipping

So the right interpretation is:

> the feasibility gate addresses the correct local failure mode and is worth keeping, but event-heavy repeated conversions still need additional downstream stabilization beyond this gate alone.

## Recommendation

Keep:

- `gp_to_beta_feasibility_gate_enabled = 1`
- `gp_to_beta_allow_seed_amplitude_scaling = 1`
- `gp_to_beta_reject_if_infeasible = 1`

Keep global `pre_Y_update` projection:

- optional
- default off

Next logical step:

- combine this gate with more realistic event throttling (`cooldown`, `spacing`, or lower forced-event density) before judging production event-heavy behavior
- then re-evaluate whether projection remains necessary in those less pathological repeated-event runs
