# Step 25 Rate / Statistics Calibration

## Scope

This step calibrates **event rates and event statistics only** for production-like `gp_zone` runs.

What was intentionally left unchanged:

- `storage_exact`
- transport PDE
- elastic solver
- conversion operator
- feasibility gate
- throttling logic
- projection default
- `two_phase`

Fixed controls during this step:

- `y_update_mass_projection_enabled = 0`
- feasibility gate on
- throttling on
- base throttling candidate from Step 24:
  - `gp_to_beta_event_cooldown_steps = 100`
  - `gp_to_beta_event_window_steps = 100`
  - `gp_to_beta_max_events_per_window = 1`
  - `gp_to_beta_event_exclusion_radius = 0.8`
  - `gp_to_beta_min_event_spacing = 0.8`
  - `gp_to_beta_max_events_global = 10`

## Files changed

- `/Users/heng/Documents/GitHub/CUDA_STO_PF/run_step25_rate_statistics_calibration.sh`
- `/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP25_RATE_STATISTICS_CALIBRATION_REPORT.md`

Remote summary CSV:

- `/tmp/step25_rate_statistics_calibration/final_key_summary.csv`

Remote run root:

- `/tmp/step25_rate_statistics_calibration/runs`

## Scan setup

Base system:

- grid: `64^3`
- raw matrix background: `xB_alpha = 0.05`
- `T = 27 C`
- `dt = 1e-5`

Scan order:

1. `gp_nuc_J0`
2. `gp_nuc_check_interval`
3. `gp_to_beta_J0_site`
4. `gp_to_beta_eta_threshold`
5. `gp_to_beta_check_interval`
6. 1000-step follow-up on representative candidates

## Main results

### 1. GP nucleation prefactor scan

200-step scan, with `gp_nuc_check_interval = 1`:

| `gp_nuc_J0` | accepted GP | accepted beta | drift | `xB` clips |
|---|---:|---:|---:|---:|
| `1e40` | 200 | 2 | 8.7491 | 1,658,637 |
| `1e38` | 200 | 2 | 8.7491 | 1,658,637 |
| `1e36` | 200 | 2 | 8.7491 | 1,658,637 |
| `1e34` | 200 | 2 | 8.7491 | 1,658,637 |
| `1e32` | 200 | 2 | 8.7491 | 1,658,637 |
| `1e30` | 191 | 2 | 8.8007 | 1,691,487 |

Conclusion:

- in this setup, reducing `gp_nuc_J0` alone from `1e40` down to `1e32` does essentially nothing
- even `1e30` barely moves GP event count
- the controlling knob is not the prefactor by itself

### 2. GP nucleation check-interval scan

Representative cases:

| case | accepted GP | accepted beta | drift | `xB` clips | `max_abs(dt*divJ)` |
|---|---:|---:|---:|---:|---:|
| `nucJ0_1e32_chk1_200` | 200 | 2 | 8.7491 | 1,658,637 | 5920.49 |
| `nucJ0_1e32_chk5_200` | 40 | 2 | 8.8860 | 2,096,686 | 1828.34 |
| `nucJ0_1e32_chk10_200` | 20 | 2 | 9.2580 | 2,572,185 | 2181.14 |
| `nucJ0_1e32_chk20_200` | 10 | 2 | 9.3928 | 2,437,939 | 14026.75 |
| `nucJ0_1e32_chk50_200` | 4 | 2 | 8.3325 | 1,603,864 | 748.35 |

Conclusion:

- `gp_nuc_check_interval` is the first knob that actually reduces GP event density
- it can push accepted GP events from `200` down to `4`
- but with `gp_to_beta_eta_threshold = 0.03`, the first `2` beta conversions still occur and still drive the run into the bad regime

### 3. GP->beta prefactor scan

Using the sparser GP candidate:

- `gp_nuc_J0 = 1e32`
- `gp_nuc_check_interval = 20`

200-step results:

| `gp_to_beta_J0_site` | accepted GP | accepted beta | drift | `xB` clips | `max_abs(dt*divJ)` |
|---|---:|---:|---:|---:|---:|
| `1e12` | 10 | 2 | 9.3928 | 2,437,939 | 14026.75 |
| `1e10` | 10 | 2 | 9.3928 | 2,437,939 | 14026.75 |
| `1e8`  | 10 | 2 | 9.3928 | 2,437,939 | 14026.75 |
| `1e6`  | 10 | 2 | 9.3928 | 2,437,939 | 14026.75 |
| `1e4`  | 10 | 2 | 8.9756 | 1,981,535 | 1356.83 |

Conclusion:

- lowering `gp_to_beta_J0_site` helps only when pushed very low
- `1e4` is measurably better than `1e6–1e12`
- but it still does **not** create a healthy production window by itself

### 4. GP->beta threshold scan

Using:

- `gp_nuc_J0 = 1e32`
- `gp_nuc_check_interval = 20`
- `gp_to_beta_J0_site = 1e8`

200-step results:

| `gp_to_beta_eta_threshold` | accepted GP | accepted beta | drift | `xB` clips | `max_abs(dt*divJ)` |
|---|---:|---:|---:|---:|---:|
| `0.03` | 10 | 2 | 9.3928 | 2,437,939 | 14026.75 |
| `0.05` | 10 | 0 | `-2.50e-15` | 0 | `1.54e-05` |
| `0.08` | 10 | 0 | `-2.50e-15` | 0 | `1.54e-05` |
| `0.10` | 10 | 0 | `-2.50e-15` | 0 | `1.54e-05` |
| `0.15` | 10 | 0 | `-2.50e-15` | 0 | `1.54e-05` |

This is the sharpest transition in the whole scan.

Interpretation:

- the pathological behavior is controlled primarily by **allowing conversion too early**
- once `gp_to_beta_eta_threshold` is raised from `0.03` to `0.05`, the bad beta conversions disappear
- the run immediately returns to a healthy GP-only regime:
  - no clipping
  - no NaN / Inf
  - near-machine-precision drift
  - bounded transport

### 5. GP->beta check-interval scan

Using:

- `gp_nuc_J0 = 1e32`
- `gp_nuc_check_interval = 20`
- `gp_to_beta_J0_site = 1e8`
- `gp_to_beta_eta_threshold = 0.08`

200-step results for `gp_to_beta_check_interval = 1, 5, 10, 20` are all identical:

- accepted GP = `10`
- accepted beta = `0`
- drift `~ -2.50e-15`
- clips `0`
- `max_abs(dt*divJ) ~ 1.54e-05`

Conclusion:

- once the threshold is high enough, conversion cadence no longer matters in this 200-step window

## 1000-step follow-up

### Pathological but sparse-beta candidate

`nucJ0_1e32_chk20_1000`:

- accepted GP = `50`
- accepted beta = `10`
- total drift = `9.4673`
- `xB` clips = `2,689,283`
- `max_abs(dt*divJ) = 14026.75`
- no NaN / Inf

`betaJ0_1e8_1000` gives the same bad outcome:

- accepted GP = `50`
- accepted beta = `10`
- drift = `9.4673`
- clips = `2,689,283`

Conclusion:

- simply making GP events sparser is **not enough**
- if beta conversion is still allowed at `eta_threshold = 0.03`, the first `~10` conversions are enough to recreate the bad regime

### Healthy production-like candidates

`etaThr_0p08_1000`:

- accepted GP = `50`
- accepted beta = `0`
- total drift = `-1.43e-14`
- `xB` clips = `0`
- `gp_closure_error = 2.55e-09`
- `max_abs(dt*divJ) = 1.54e-05`
- `eta_integral = 4.3866e-02`
- beta volume fraction = `0`
- no NaN / Inf

`betaChk_10_1000` is identical for the metrics that matter here:

- accepted GP = `50`
- accepted beta = `0`
- total drift = `-1.43e-14`
- `xB` clips = `0`
- `max_abs(dt*divJ) = 1.54e-05`

## Final interpretation

Step 25 confirms the diagnosis from Step 24:

- the remaining pathology was **event statistics**, not solver correctness
- once conversion statistics are made physically sparse enough, the run becomes healthy again with:
  - projection off
  - feasibility gate on
  - throttling on

But the scan also shows a limitation:

- within the requested parameter grid, the healthy window is currently a **GP-only** window
- the scanned points did **not** produce a simultaneously healthy and visibly converting `GP -> beta` regime

In other words:

1. `eta_threshold = 0.03` is too permissive and leads to pathological beta conversion even when GP events are sparse
2. `eta_threshold >= 0.05` suppresses those bad conversions and restores a clean run
3. the transition is sharp, which suggests the next useful calibration is a **finer conversion-threshold / rate scan near the boundary**, not more solver changes

## Best candidates from this scan

### Best healthy production-like candidate

- `gp_nuc_J0 = 1e32`
- `gp_nuc_check_interval = 20`
- `gp_to_beta_J0_site = 1e8`
- `gp_to_beta_eta_threshold = 0.08`
- `gp_to_beta_check_interval = 1` or `10`
- feasibility gate on
- throttling on
- projection off

Outcome:

- accepted GP events: `50 / 1000`
- accepted beta events: `0 / 1000`
- no NaN / Inf
- no clipping
- drift near machine precision
- bounded transport

### Best sparse-but-still-bad candidate

- `gp_nuc_J0 = 1e32`
- `gp_nuc_check_interval = 20`
- `gp_to_beta_eta_threshold = 0.03`

Outcome:

- accepted GP events: `50 / 1000`
- accepted beta events: `10 / 1000`
- still pathological

This is the strongest evidence that **beta conversion statistics**, not GP density alone, are the decisive remaining control.

## Recommended next step

Do **not** change the solver.

The next clean calibration step should be a **fine conversion-boundary scan**, for example:

- `gp_to_beta_eta_threshold = 0.035, 0.040, 0.045, 0.050`
- optionally combine with
  - `gp_to_beta_J0_site = 1e4, 1e5, 1e6`
  - `gp_to_beta_check_interval = 5, 10, 20`

Reason:

- Step 25 already found the healthy side and the pathological side
- what is missing is the boundary where **sparse, visible beta conversion** starts without reintroducing clipping/drift

## Bottom line

Step 25 succeeds in its main purpose:

- it shows that reducing event rates/statistics can cure the remaining pathology
- it confirms the solver chain is not the root problem
- it identifies the conversion threshold as the dominant event-statistics control

What it does **not** yet deliver is a final production setting with both:

- healthy dynamics
- and nonzero beta conversion

That final setting likely sits near the `eta_threshold` boundary just above `0.03`, and needs a finer scan than the coarse grid requested here.

