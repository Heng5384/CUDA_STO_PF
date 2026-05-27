# Step 32 Single Conservative `eta=1.0` Validation

## Purpose

This run isolates a single representative mature GP case:

- `model_mode = gp_zone`
- `gp_y_update_mode = conservative_y_rhs`
- `xB_background = 0.03`
- `R_GP = 1 nm`
- `eta_peak = 1.0`
- `depletion_radius_factor = 2`
- `grid = 96^3`
- `dx = 0.1 nm`
- `dt = 1e-5`
- `steps = 5000`

The question is narrow:

> Can `conservative_y_rhs` run this representative mature GP case to 5000 steps without the early `storage_exact` blow-up?

## Initialization

Initialization uses the offline smooth self-consistent profile bundle:

- `phi = 0`
- smooth `eta(r)`
- smooth mass-conserving `xB_alpha(r)`
- diagnostic `xBtot_gp`

No stochastic nucleation, no GP->beta conversion, no projection.

## Required Diagnostics

The summary CSV records:

- `survival_time_steps`
- `eta_max_final`
- `eta_integral_final`
- `V_h_proxy_final`
- `xB_min_final`, `xB_max_final`
- `xB_clip_count_high_total`, `xB_clip_count_low_total`
- `total_relative_drift_final`
- `gp_closure_error_final`
- `max_abs_dt_divJ`
- `max_abs_fY`
- `min_h_alpha`
- `min_h_alpha_q`
- `max_abs_lagged_dYdt`
- `cumulative_mass_drift_slope_tail`
- `NaN_or_Inf`
- `classification`

## Output

- Script:
  - [/Users/heng/Documents/GitHub/CUDA_STO_PF/run_step32_single_conservative_eta1_validation.sh](/Users/heng/Documents/GitHub/CUDA_STO_PF/run_step32_single_conservative_eta1_validation.sh)
- Summary CSV:
  - `/tmp/step32_single_conservative_eta1_validation/final_key_summary.csv`

## Interpretation Rule

- `stable_or_slow_relaxation`
  - full 5000-step run
  - no `NaN/Inf`
  - no `xB` clipping to `0/1`
  - bounded drift and bounded `dt*divJ`

- `marginal_but_bounded`
  - full run, but drift or transport stress remains noticeable

- `unstable`
  - `NaN/Inf`, clipping, or early termination
