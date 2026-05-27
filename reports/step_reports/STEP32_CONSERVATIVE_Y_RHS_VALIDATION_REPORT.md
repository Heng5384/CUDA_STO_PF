# Step 32: `conservative_y_rhs` Validation Report

## Purpose

This step adds a new optional

- `gp_y_update_mode = conservative_y_rhs`

for `model_mode = gp_zone`.

It is intended to avoid the algebraic `storage_exact` reconstruction

\[
xB_\alpha = \frac{xB_{tot}^{target} - h_{GP}xB_{GP} - h_\beta}{h_\alpha}
\]

in the `h_\alpha \to 0` limit by updating `Y = logit(xB_alpha)` directly with a paper-style conservative source-term equation.

This step does **not** remove or replace `storage_exact`. It adds a new mode only.

## Implemented Equation

Starting from the GP storage identity

\[
xB_{tot} = h_\alpha xB_\alpha + h_{GP} xB_{GP} + h_\beta
\]

and

\[
xB_\alpha = \sigma(Y), \qquad q = xB_\alpha(1-xB_\alpha),
\]

the new mode uses the conservative source form

\[
f_Y
=
\operatorname{div}J
- \dot{h}_{GP}(xB_{GP}-xB_\alpha)
- \dot{h}_{\beta}(1-xB_\alpha)
- \bar{D}_Y \nabla^2 Y
- (h_\alpha q - 1)\,\dot{Y}_{lagged}.
\]

Then the Fourier update is

\[
Y_k^{n+1}
=
\frac{Y_k^n + \Delta t\,f_{Y,k}}
{1 + \Delta t\,\bar{D}_Y k^2}.
\]

After inverse FFT:

1. normalize by `1/N`
2. clamp `Y` to existing safe bounds
3. update `xB_alpha = sigmoid(Y)`

This mode **does not** reconstruct `xB_alpha` algebraically from `xBtot_target / h_alpha`.

## Files Changed

- [/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.h](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.h)
- [/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu)
- [/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu)
- [/Users/heng/Documents/GitHub/CUDA_STO_PF/pf_params.h](/Users/heng/Documents/GitHub/CUDA_STO_PF/pf_params.h)
- [/Users/heng/Documents/GitHub/CUDA_STO_PF/run_step32_conservative_y_rhs_validation.sh](/Users/heng/Documents/GitHub/CUDA_STO_PF/run_step32_conservative_y_rhs_validation.sh)

## Diagnostics Added

The new mode extends the Y-RHS diagnostics with:

- `mean_h_alpha_q_for_Y_rhs`
- `min_h_alpha_for_Y_rhs`
- `min_h_alpha_q_for_Y_rhs`
- `max_h_alpha_q_for_Y_rhs`
- `max_abs_fY`
- `max_abs_lagged_dYdt`

Together with existing diagnostics:

- `mean_xBtot_gp_before_Y`
- `mean_xBtot_gp_after_Y_update`
- `total_relative_drift`
- `gp_closure_error`
- `xB_clip_count_high/low`
- `dt_divJ_min/max`
- `delta_mass_Y_update`
- `delta_Y_k0_re`

## Validation Matrix

The validation script runs:

1. `two_phase` regression
   - `two_phase_ref_200`
   - `two_phase_conservative_flag_200`

2. `gp_zone, eta = 0` regression
   - `gp_zone_eta0_old_rhs_200`
   - `gp_zone_eta0_conservative_200`

3. Step 26B-style tiny GP seed stability
   - `gp_zone_tinyseed_conservative_5000`

4. Mature `eta_peak = 1.0` profile-bundle comparison
   - `profilebundle_storage_exact_xB03_R1_peak1p0_dep2_N96_5000`
   - `profilebundle_conservative_y_rhs_xB03_R1_peak1p0_dep2_N96_5000`
   - `profilebundle_storage_exact_xB03_R1_peak1p0_dep5_N96_5000`
   - `profilebundle_conservative_y_rhs_xB03_R1_peak1p0_dep5_N96_5000`

## Runtime Outputs

The validation script writes:

- `/tmp/step32_conservative_y_rhs_validation/final_key_summary.csv`

and per-case diagnostics under:

- `/tmp/step32_conservative_y_rhs_validation/runs/...`

## Evaluation Questions

The intended interpretation is:

1. Does `two_phase` remain unchanged when the new GP mode string is present but unused?
2. Does `gp_zone, eta=0` remain close to the old non-singular behavior?
3. Does the tiny-GP-seed case remain stable?
4. In the `eta_peak = 1.0` mature profile-bundle case, does `conservative_y_rhs`
   - reduce `xB` clipping,
   - reduce `max_abs(dt*divJ)`,
   - reduce `total_relative_drift`,
   - and avoid the `h_alpha -> 0` algebraic blow-up seen with `storage_exact`?

## Current Status

Code changes are in place.

In the current local environment, `nvcc` is not available, so this report and the validation script are prepared but not compiled or executed here. The expected next step is to run:

```bash
bash /Users/heng/Documents/GitHub/CUDA_STO_PF/run_step32_conservative_y_rhs_validation.sh
```

on a CUDA-capable workstation or cluster node.
