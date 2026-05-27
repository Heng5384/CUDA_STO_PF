# STEP38C GP Eta STO Parameterization Alignment

## Purpose

This step aligns the **parameterization logic and diagnostics** of the GP `eta`
field with the existing `phi` thin-interface pseudobinary STO framework.

No evolution equation was changed in this step.

## Unified framework

Both `phi` and `eta` are now described using the same STO-type pseudobinary
reaction phase-field parameterization workflow:

1. physical inputs:
   - interface energy `gamma_i`
   - interface width `lambda_i`
   - diffusion / mobility reference inputs
2. physical thin-interface quantities:
   - `W_i_phys`
   - `kappa_i_phys`
   - `L_i_phys`
3. code-unit quantities used inside kernels:
   - `W_i_code = W_i_phys / w_ref`
   - `kappa_i_code = kappa_i_phys / (w_ref * dx_ref^2)`
   - `L_i_code = L_i_phys * w_ref * t0_diff`

The difference between `phi` and `eta` is therefore **only the physical input
values**, not the nondimensionalization methodology.

## Shared scales

`phi` and `eta` both use the same runtime reference scales:

- `w_ref`
- `dx_ref`
- `t0_diff`

No eta-specific energy scale, length scale, or time scale is introduced.

## Phi side

For `phi` (beta-Ag2Te line-compound representation), the code already uses
code-unit parameters in the kernel:

- `W`
- `kappa_phi`
- `L_phi`

The diagnostics now explicitly reconstruct and print:

- `gamma_alpha_beta`
- `lambda_phi`
- `W_phi_phys`
- `kappa_phi_phys`
- `L_phi_phys`
- `W_phi_code`
- `kappa_phi_code`
- `L_phi_code`

## Eta side

For `eta` (GP zone order parameter), the kernel now also consistently uses
code-unit parameters:

- `P.gp_W_eta`
- `P.gp_kappa_eta`

Input handling now supports:

- physical input keys:
  - `gp_W_eta_phys`
  - `gp_kappa_eta_phys`
- code input keys:
  - `gp_W_eta_code`
  - `gp_kappa_eta_code`
- legacy keys:
  - `gp_W_eta`
  - `gp_kappa_eta`

Legacy keys are not handled silently:

- if magnitudes look physical, they are converted to code units with a warning
- if magnitudes look like code units, they are used as-is with a warning

## STO-type GP kinetic references

The GP `eta` field is **not** presented as a separate phenomenological
relaxation scheme. Instead, its diagnostics are labeled as:

- STO GP reaction/interface reference
- STO GP diffusion-limit reference

At present:

- the diffusion-limit reference is computed
- the finite reaction/interface reference remains unavailable unless an
  explicit GP interface mobility model is supplied and implemented

If unavailable, runtime diagnostics print:

`STO GP reaction/interface reference unavailable: M_GP not specified or finite-M formula not implemented`

## Important interpretation note

The GP diffusion-limit value is only a **STO-type kinetic reference**.
It is not automatically the "correct" GP mobility, because GP is modeled here
as a precursor / order field rather than necessarily a true fixed-stoichiometry
compound phase.

It should be used as a diagnostic scale for comparison with the manually chosen
`gp_L_eta`, not as an automatic replacement.
