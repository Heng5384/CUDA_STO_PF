# Step 24 Event Throttling Calibration

## Scope

This step calibrates **event throttling only** for production-like `gp_zone` runs with:

- stochastic GP nucleation enabled
- stochastic `GP -> beta` conversion enabled
- feasibility gate enabled
- `y_update_mass_projection_enabled = 0`

What was intentionally left unchanged:

- `storage_exact`
- transport PDE
- elastic solver
- conversion operator
- feasibility gate logic
- projection default
- `two_phase`

## Files changed

- `/Users/heng/Documents/GitHub/CUDA_STO_PF/run_step24_event_throttling_calibration.sh`
- `/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP24_EVENT_THROTTLING_CALIBRATION_REPORT.md`

## Scan setup

Primary scan:

- grid: `64^3`
- background: uniform raw-field matrix, `xB_alpha = 0.05`
- `T = 27 C`
- `dt = 1e-5`
- projection: off
- feasibility gate: on

Aggressive stochastic rates used for this calibration stress test:

- `gp_nuc_enabled = 1`
- `gp_nuc_check_interval = 1`
- `gp_nuc_J0 = 1e40`
- `gp_nuc_gamma = 0`
- `gp_to_beta_enabled = 1`
- `gp_to_beta_check_interval = 1`
- `gp_to_beta_stochastic_enabled = 1`
- `gp_to_beta_J0_site = 1e12`
- `gp_to_beta_gamma = 0`
- `gp_to_beta_drive_mode = constant`
- `gp_to_beta_drive_const = 1`

Output root:

- `/tmp/step24_event_throttling_validation/runs`

Remote summary CSV:

- `/tmp/step24_event_throttling_validation/final_key_summary.csv`

## Parameter sweep

Coarse 200-step cases:

- no throttle reference
- cooldown: `20`, `50`, `100`
- exclusion radius / min spacing: `0.4`, `0.6`, `0.8`, `1.0`
- max events per window: `1`, `2`, `5`
- event window steps: `20`, `50`, `100`
- global cap: `5`, `10`, `20`
- two strict combinations:
  - `strict_combo1`: `cooldown=100`, `radius=1.0`, `window=1/100`, `global=5`
  - `strict_combo2`: `cooldown=100`, `radius=0.8`, `window=1/100`, `global=10`

1000-step follow-up:

- `cooldown_50_1000`
- `cooldown_100_1000`
- `strict_combo2_1000`

## Key results

### Healthy reference

`nuc_only_ref_200`:

- accepted GP events: `200`
- accepted `GP -> beta` events: `0`
- `xB` clips: `0`
- total relative drift: `-2.64e-15`
- `gp_closure_error`: `2.55e-09`
- `max_abs(dt*divJ)`: `1.55e-05`
- no NaN / Inf

This confirms the GP nucleation-only path is still healthy.

### 200-step coarse scan

Representative cases:

| case | accepted beta | total `xB` clips | total drift | mean spacing | max `|dt*divJ|` |
|---|---:|---:|---:|---:|---:|
| `ref_no_throttle_200` | 200 | 1,784,886 | 8.7085 | 0.644 | 442.18 |
| `base_c20_r04_w1_t20_g20_200` | 10 | 2,424,764 | 9.3564 | 1.944 | 5920.49 |
| `cooldown_50_200` | 4 | 1,591,643 | 8.6742 | 3.299 | 5920.49 |
| `cooldown_100_200` | 2 | 1,658,637 | 8.7491 | 7.285 | 5920.49 |
| `strict_combo2_200` | 2 | 1,658,637 | 8.7491 | 7.285 | 5920.49 |

Observed pattern:

- The only throttling knobs that materially changed outcomes were **longer cooldown** and the equivalent **longer one-event window**.
- Exclusion radius by itself did little.
- Raising `max_events_per_window` above `1` did not help.
- Global caps alone were not useful in this regime.

Best 200-step throttled case in this scan:

- `cooldown_50_200`
- accepted `GP -> beta` events: `4`
- total clips reduced relative to no-throttle
- drift reduced slightly relative to no-throttle

### 1000-step follow-up

| case | accepted GP | accepted beta | rejected cooldown | rejected spacing | rejected window | rejected global | total `xB` clips | total drift | `eta_integral` | beta volume fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cooldown_50_1000` | 1000 | 20 | 979 | 9 | 380 | 48 | 1,917,836 | 8.9295 | 0.69306 | 0.0013388 |
| `cooldown_100_1000` | 1000 | 10 | 990 | 4 | 190 | 0 | 1,857,041 | 8.8672 | 0.69344 | 0.0006566 |
| `strict_combo2_1000` | 1000 | 10 | 990 | 37 | 990 | 99 | 1,857,041 | 8.8672 | 0.69344 | 0.0006566 |

Additional observations:

- no NaN / Inf in any 1000-step follow-up case
- `gp_closure_error` stayed small:
  - `cooldown_50_1000`: `8.48e-07`
  - `cooldown_100_1000`: `3.06e-07`
  - `strict_combo2_1000`: `3.06e-07`
- `max_abs(dt*divJ)` remained extremely large in all conversion-enabled follow-ups:
  - about `5.92e3`

Interpretation:

- Moving from `cooldown=50` to `cooldown=100` cuts accepted beta events from `20` to `10` and slightly lowers clipping and drift.
- `strict_combo2_1000` produced **the same accepted-event trajectory and final fields** as `cooldown_100_1000` for the metrics that matter here. The extra spacing/window/global constraints changed rejection bookkeeping, but did not improve the realized dynamics further.

## Production-like window assessment

Against the requested success criteria:

- no NaN / Inf: **pass**
- projection not required: **pass**
- no repeated local conversion burst: **partially pass**
  - cooldown/window throttling clearly suppresses bursts
  - accepted event count drops from `200` to `10–20`
- accepted event count physically reasonable: **partially pass**
  - `10` accepted conversions in `1000` steps is much more realistic than `200`
- `xB` clipping near zero or strongly bounded: **fail**
  - clipping remains of order `1.8e6`
- total drift acceptable: **fail**
  - drift remains about `8.87–8.93`

## Main conclusion

Step 24 shows that **throttling alone helps event statistics but does not create a clean production window** under this very aggressive stochastic-rate setup.

What the scan does establish:

1. The remaining pathology is **not** a `storage_exact`, transport, elastic, or conversion-operator bug.
2. Cooldown / one-event window controls are the dominant throttling knobs.
3. `cooldown = 100`, `window_steps = 100`, `max_events_per_window = 1` is the best throttling regime found in this scan.
4. Extra spacing / global-cap constraints on top of that regime did not materially improve the realized trajectory.

## Recommended calibrated candidate

Best throttling candidate from this scan:

- `gp_to_beta_event_cooldown_steps = 100`
- `gp_to_beta_event_window_steps = 100`
- `gp_to_beta_max_events_per_window = 1`
- `gp_to_beta_event_exclusion_radius = 0.4` to `0.8`
- `gp_to_beta_min_event_spacing = same as exclusion radius`
- `gp_to_beta_max_events_global = 10` or large

Why:

- it cuts accepted `GP -> beta` events to `10 / 1000`
- it avoids repeated local conversion bursts much better than weaker throttling
- it does not introduce NaN / Inf
- it keeps feasibility-gated event compensation healthy

But this is only a **best available throttling candidate**, not yet a fully acceptable production recipe.

## What is still missing

Under the current stress-test rates, accepted GP nucleation remains `1000 / 1000` and the transport field still becomes extremely aggressive after long evolution. That means the next calibration step should not be more throttling alone. It should be **rate/statistics calibration**, for example:

- reduce `gp_nuc_J0`
- reduce `gp_to_beta_J0_site`
- increase `gp_to_beta_eta_threshold`
- increase `gp_to_beta_check_interval`
- reduce GP event density before conversion is even considered

That is the clean next step if the target is truly production-like event statistics without relying on projection.

