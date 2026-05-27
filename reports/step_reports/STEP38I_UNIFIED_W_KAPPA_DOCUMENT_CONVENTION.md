# STEP38I: Unified phi/eta W-kappa document convention

## Adopted convention

This step adopts the document convention for **both** `phi` and `eta`:

\[
W = \frac{12 \gamma}{\lambda}, \qquad \kappa = \frac{3}{2}\gamma\lambda
\]

This follows the document identities:

\[
\kappa W = 18 \gamma^2, \qquad \frac{\kappa}{W} = \left(\frac{\lambda}{8}\right)^2
\]

## Unified nondimensionalization

Both `phi` and `eta` now use the same nondimensionalization:

\[
W_i^{code} = \frac{W_i^{phys}}{w_{ref}}, \qquad
\kappa_i^{code} = \frac{\kappa_i^{phys}}{w_{ref} dx_{ref}^2}
\]

with the same global reference scales:

- `w_ref`
- `dx_ref`
- `t0_diff`
- `mu_reference_scale`

## Phi vs eta

The difference between `phi` and `eta` is **only** in physical inputs:

- `phi`: `gamma_alpha_beta`, `lambda_sm`
- `eta`: `gp_gamma_alpha_gp`, `gp_l_eta_nm`

They no longer differ by using different W/kappa generation formulas.

## What changed

- `eta` interface-parameter helper functions were changed from the old `ln(9)` convention to the document convention.
- `eta` diagnostics were updated to print:
  - document-convention physical values
  - document-convention code values
  - kernel-used values
  - kernel/document ratios
- `phi` diagnostics were updated to print the same expected-vs-kernel comparison.

## What did not change

This step does **not** change:

- the `phi` PDE
- the `eta` PDE
- the `Y/xB` PDE
- chemical-potential scaling
- `mu_reference_scale`
- `w_ref`
- `dx_ref`
- `t0_diff`
- `gp_L_eta`
- `L_phi`

This is a parameter-generation and diagnostic-consistency cleanup only.

## Important note

Changing the W/kappa convention changes:

- eta stiffness,
- eta semi-implicit damping,
- the STO diffusion-limit reference for eta.

So any earlier `gp_L_eta` or `f_eta` baseline must be **rescanned** after this step.
