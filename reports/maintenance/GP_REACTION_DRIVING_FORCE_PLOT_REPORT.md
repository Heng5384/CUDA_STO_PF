# GP Reaction Driving Force Plot Report

## Scope

This report plots and interprets the **GP-zone stoichiometric reaction chemical driving force**

\[
-\Delta \mu_r^{GP}(x_B,T)
= \nu_A \mu_A^\alpha(x_B,T) + \nu_B \mu_B^\alpha(x_B,T) - \mu_{GP}^0(T)
\]

with:

- `nu_A = 0.65`
- `nu_B = 0.35`
- `xB_GP = 0.35`
- `Scheme A: mu_GP^0(T) = g_alpha(xB_GP, T)`

No solver code was changed.

## Files

- CSV: [gp_reaction_drive_table.csv](/Users/heng/Documents/GitHub/CUDA_STO_PF/gp_reaction_drive_table.csv)
- Raw plot: [gp_reaction_drive_raw.png](/Users/heng/Documents/GitHub/CUDA_STO_PF/gp_reaction_drive_raw.png)
- Code-used plot: [gp_reaction_drive_code.png](/Users/heng/Documents/GitHub/CUDA_STO_PF/gp_reaction_drive_code.png)
- Raw vs code: [gp_reaction_drive_raw_vs_code.png](/Users/heng/Documents/GitHub/CUDA_STO_PF/gp_reaction_drive_raw_vs_code.png)
- Reaction vs diffusion potential: [gp_reaction_drive_vs_diffusion_potential.png](/Users/heng/Documents/GitHub/CUDA_STO_PF/gp_reaction_drive_vs_diffusion_potential.png)
- Optional temperature comparison: [gp_reaction_drive_temperature_comparison.png](/Users/heng/Documents/GitHub/CUDA_STO_PF/gp_reaction_drive_temperature_comparison.png)

## Thermodynamic Definitions Used

### Raw regular-solution version

The raw branch uses the unmodified CALPHAD/regular-solution chemical potentials:

- `mu_PbTe_calphad(T, xB)`
- `mu_Ag2Te_calphad(T, xB)`

from [thermo_utils.h](/Users/heng/Documents/GitHub/CUDA_STO_PF/thermo_utils.h:342).

### Code-used version

The code-used branch uses:

- `mu_PbTe_raw(T, xB)`
- `mu_Ag2Te_raw(T, xB)`

from [thermo_utils.h](/Users/heng/Documents/GitHub/CUDA_STO_PF/thermo_utils.h:388).

In the current code path:

- `thermo_convex_extrapolation_enabled = 1` by default in [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:4793)
- the convex-extrapolation threshold is `X_LIMIT_CONVEX = 0.09` in [thermo_utils.h](/Users/heng/Documents/GitHub/CUDA_STO_PF/thermo_utils.h:266)

That matters here because `xB_GP = 0.35` is deep inside the convexified branch for the code-used thermodynamics.

## Direct Answers

### 1. At `xB = 0.03`, is `-Delta_mu_r_GP` positive or negative?

At `T = 653.15 K`:

- raw regular-solution:
  - `-Delta_mu_r_GP(0.03) = -2.434676014945e+02 J/mol`
  - negative
- code-used thermodynamics:
  - `-Delta_mu_r_GP_code(0.03) = -1.681142252747e+03 J/mol`
  - negative

So at `xB = 0.03`, the GP reaction driving force is **negative** in both branches.

### 2. At `xB = 0.05`, is `-Delta_mu_r_GP` positive or negative?

At `T = 653.15 K`:

- raw regular-solution:
  - `-Delta_mu_r_GP(0.05) = 2.890683634217e+02 J/mol`
  - positive
- code-used thermodynamics:
  - `-Delta_mu_r_GP_code(0.05) = -1.148606287831e+03 J/mol`
  - negative

So at `xB = 0.05`:

- raw regular-solution says **positive**
- code-used thermodynamics says **negative**

That sign flip is the most important outcome of this plot set.

### 3. How large is the difference between raw regular-solution and code-used thermodynamics?

It is small for `mu_A(xB)` and `mu_B(xB)` as long as the **background** `xB < 0.09`, because both branches are identical there.

It becomes large in the GP reaction driving force because `mu_GP0(T)` is evaluated at `xB_GP = 0.35`, which is:

- in the physical branch for the raw regular-solution definition
- in the convex-extrapolated branch for the code-used definition

Numerically at `T = 653.15 K`:

- `mu_GP0_raw = -1.503513949839e+05 J/mol`
- `mu_GP0_code = -1.489137203327e+05 J/mol`

Difference:

- `mu_GP0_code - mu_GP0_raw = 1.437674651171e+03 J/mol`

That shift is large enough to change the sign of `-Delta_mu_r_GP` near the `xB = 0.03–0.05` backgrounds you care about.

### 4. The current GP eta RHS should use `-Delta_mu_r_GP`, not `mu_B - mu_A`

Agreed.

`mu_B - mu_A` is the **diffusion potential**. It is not the same quantity as the stoichiometric GP reaction driving force

\[
\nu_A \mu_A^\alpha + \nu_B \mu_B^\alpha - \mu_{GP}^0
\]

The comparison plot [gp_reaction_drive_vs_diffusion_potential.png](/Users/heng/Documents/GitHub/CUDA_STO_PF/gp_reaction_drive_vs_diffusion_potential.png) shows that these two curves have different magnitude, different structure, and can even imply different sign logic for growth tendency.

So if the GP phase evolution is intended to represent a stoichiometric reaction channel, then the thermodynamically relevant driving force is the **reaction** quantity, not the diffusion potential.

### 5. If the code now uses `mu_B-mu_A` or `g_GP-g_alpha`, where would it need modification?

The current `eta` RHS is not using `mu_B - mu_A` directly. It is using a **free-energy density difference**:

- `g_alpha = (1-xB_alpha) * muA_alpha + xB_alpha * muB_alpha`
- `g_gp = (1-xB_gp) * muA_gp + xB_gp * muB_gp - gp_delta_g0`
- `dgbulk_deta ~ (g_gp - g_alpha)`

This is implemented in [cuda_kernels.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:801) inside `compute_eta_rhs_kernel(...)`.

That is the first place that would need modification if you want the solver to use the explicit stoichiometric reaction driving force

\[
\nu_A \mu_A^\alpha(x_B) + \nu_B \mu_B^\alpha(x_B) - \mu_{GP}^0
\]

instead of `g_gp - g_alpha`.

The host-side diagnostic helper that mirrors this GP nucleation driving concept is:

- [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:5429)

inside `compute_delta_g_nuc_GP_host(...)`.

That helper would also need to be updated if you want debug output and reported thermodynamic diagnostics to stay consistent with a changed solver RHS.

## Interpretation

## What the plots show

### Raw regular-solution branch

- `-Delta_mu_r_GP` crosses from negative to positive between `xB = 0.03` and `xB = 0.05`
- this means the raw regular-solution model places the GP reaction threshold near that background range

### Code-used branch

- `-Delta_mu_r_GP_code` stays negative at both `xB = 0.03` and `xB = 0.05`
- the reason is not local-background convexification
- the reason is that `mu_GP0_code = g_alpha_code(xB_GP=0.35)` is evaluated in the convexified high-`xB` branch

So the present code-used thermodynamics effectively penalizes the `xB_GP = 0.35` reference state much more strongly than the raw regular-solution model.

## Implication for GP kinetics

If your physical intention is:

- a stoichiometric GP reaction channel with fixed `nu_A = 0.65`, `nu_B = 0.35`

then the correct scalar driving metric is the reaction chemical driving force plotted here, not the diffusion potential.

If your numerical implementation continues to evolve `eta` using `g_gp - g_alpha`, then it is not identical to the requested reaction formula, because:

- `g_alpha(xB)` uses local composition weights `(1-xB, xB)`
- your target reaction uses fixed stoichiometric weights `(0.65, 0.35)`

Those are different constructions.

## Practical Conclusion

For the current model at `T = 653.15 K`:

- `xB = 0.03` does **not** provide positive GP reaction driving force under either branch
- `xB = 0.05` provides positive driving force only in the raw regular-solution branch
- the current code-used convexified thermodynamics makes the `xB_GP = 0.35` reference state substantially less favorable

So if GP growth is being judged from `mu_B - mu_A`, or even from `g_gp - g_alpha`, that should not be conflated with the stoichiometric reaction driving force requested here.

## Recommended next step

Do not change the solver yet.

First decide which thermodynamic definition is intended for the GP reference state:

1. raw regular-solution `mu_GP0 = g_alpha_raw(0.35, T)`, or
2. code-used convexified `mu_GP0 = g_alpha_code(0.35, T)`

Only after that should the GP `eta` RHS be refactored, because the sign and magnitude of the reaction driving force near `xB = 0.03–0.05` depends strongly on that choice.
