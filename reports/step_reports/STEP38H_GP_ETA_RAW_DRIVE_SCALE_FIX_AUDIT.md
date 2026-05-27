# STEP38H: GP eta active raw reaction drive scale fix audit

## Scope

This step fixed the **chemical-potential scaling only** in the active `gp_raw_reaction_drive_only=1` branch for GP eta.  
We did **not** change:

- the eta RHS structure,
- the phi RHS,
- the Y/xB equation,
- `gp_W_eta` / `gp_kappa_eta` conversion,
- `gp_L_eta`,
- update order.

## Modified files

- [/Users/heng/Documents/GitHub/CUDA_STO_PF/pf_params.h](/Users/heng/Documents/GitHub/CUDA_STO_PF/pf_params.h)
- [/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.h](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.h)
- [/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu)
- [/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu)

## Exact before formula

Before, the active raw branch in `compute_eta_rhs_kernel()` used raw chemical potentials in J/mol directly:

```cpp
double muA_alpha_raw = mu_PbTe_calphad(temperature_K, xB_alpha);
double muB_alpha_raw = mu_Ag2Te_calphad(temperature_K, xB_alpha);
double minus_delta_mu_r_gp =
    gp_reaction_nu_A * muA_alpha_raw +
    gp_reaction_nu_B * muB_alpha_raw -
    gp_mu_reference_raw;
dgbulk_deta = c_ref * (1.0 - h_phi) * hp_eta * (-minus_delta_mu_r_gp);
```

This mixed a raw J/mol-scale chemical drive with code-unit `gp_W_eta` / `gp_kappa_eta` in the same eta RHS.

## Exact after formula

Now the same branch first converts all chemical potentials using the **same** `mu_reference_scale` used by phi:

```cpp
double muA_alpha_raw = mu_PbTe_calphad(temperature_K, xB_alpha);
double muB_alpha_raw = mu_Ag2Te_calphad(temperature_K, xB_alpha);
const double mu_scale =
    (fabs(mu_reference_scale) > 1.0e-300) ? mu_reference_scale : 1.0;
const double muA_alpha_dimless = muA_alpha_raw / mu_scale;
const double muB_alpha_dimless = muB_alpha_raw / mu_scale;
const double gp_mu_reference_dimless = gp_mu_reference_raw / mu_scale;
double minus_delta_mu_r_gp =
    gp_reaction_nu_A * muA_alpha_dimless +
    gp_reaction_nu_B * muB_alpha_dimless -
    gp_mu_reference_dimless;
dgbulk_deta = c_ref * (1.0 - h_phi) * hp_eta * (-minus_delta_mu_r_gp);
```

The raw-unit behavior is still available only behind:

- `gp_raw_reaction_drive_use_raw_units_debug=1`

and prints a strong warning when enabled.

## Compile status

workstation compile succeeded with the known stable command:

```bash
make clean >/dev/null 2>&1 || true
make NVCC=/usr/local/cuda-12.9/bin/nvcc \
     NVCCFLAGS="-arch=sm_120 -O2 -std=c++14 --expt-relaxed-constexpr -Xcompiler -O1 -Xcompiler -Wno-format-truncation" \
     -j$(nproc)
```

## Rerun outputs

Baseline rerun directory on workstation:

- `/home/zhiheng/PF/CUDA_STO_PF/Results/ch_T380_cuda_96x96x96_dt0.0001_steps100_xB0.030/step38h_gp_eta_rhs_attr_f1em5_scaled`

Copied local outputs:

- per-step attribution CSV:
  - [/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/data/step38h_eta_raw_scale_fix/step38h_phi_eta_rhs_attribution_per_step.csv](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/data/step38h_eta_raw_scale_fix/step38h_phi_eta_rhs_attribution_per_step.csv)
- max-location audit CSV:
  - [/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/data/step38h_eta_raw_scale_fix/step38h_eta_bulk_max_location.csv](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/data/step38h_eta_raw_scale_fix/step38h_eta_bulk_max_location.csv)
- run log:
  - [/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/data/step38h_eta_raw_scale_fix/step38h_run.log](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/data/step38h_eta_raw_scale_fix/step38h_run.log)

Before/after comparison CSV:

- [/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/data/step38h_eta_raw_scale_fix/step38h_before_after_comparison.csv](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/data/step38h_eta_raw_scale_fix/step38h_before_after_comparison.csv)

## Eta-vs-phi unit alignment table

| Item | phi | eta after Step38H | Aligned with phi? |
|---|---|---|---|
| chemical scale | `mu_A_dimless`, `mu_B_dimless` using `mu_reference_scale` | raw branch now divides `mu_PbTe_calphad`, `mu_Ag2Te_calphad`, and `gp_mu_reference_raw` by the same `mu_reference_scale` | Yes |
| W scale | `W` is kernel-used code unit | `gp_W_eta` is kernel-used code unit | Yes |
| kappa scale | `kappa_phi` is kernel-used code unit | `gp_kappa_eta` is kernel-used code unit | Yes |
| L scale | `L_phi` is code-unit mobility | `gp_L_eta` is code-unit mobility | Yes |
| RHS sum scale | `phi_rhs = chem + W*g'(phi) + elastic` all in code units | `eta_rhs = dgbulk_deta + gp_W_eta*g'(eta) + elastic` all in code units | Yes |
| semi-implicit denominator scale | `1 + L_phi*dt*kappa_phi*k2` | `1 + gp_L_eta*dt*gp_kappa_eta*k2` | Yes |

## Max-location audit findings

At the max-`|dgbulk_deta|` location in the baseline rerun:

- region: `interface`
- `eta_before = 4.9266013698e-01`
- `phi_before = 4.9256461386e-01`
- `xB_before = 2.8405822365e-02`

Raw vs dimless at the same location:

- `drive_raw_formula_value = 2.8664768755e+03`
- `dgbulk_deta_raw_formula_value = -2.7610504483e+03`
- `energy_scale = 1.3779024000e+04`
- `drive_dimless_formula_value = 2.0803192414e-01`
- `dgbulk_deta_dimless_formula_value = -2.0038069811e-01`
- `dgbulk_deta_kernel_used = -2.0043664955e-01`

Checks:

- `dgbulk_deta_raw_formula_value / dgbulk_deta_dimless_formula_value = 1.37790239995e+04`
- this matches `energy_scale = 1.3779024000e+04`
- `dgbulk_deta_kernel_used` matches the **dimensionless** formula, not the raw formula

## Before/after comparison

Known before values from Step38G/38F:

- `eta_bulk_rhs_absmax ≈ 2.914e3`
- `phi_chem_rhs_absmax ≈ 1.478`
- `eta_bulk_rhs_absmax / phi_chem_rhs_absmax ≈ 1.97e3`
- `eta_Ldt_bulk_absmax / phi_Ldt_chem_absmax ≈ 0.39 mean, ≈ 0.47 max`

After Step38H baseline rerun:

- `eta_bulk_rhs_absmax = 2.1135036968e-01`
- `phi_chem_rhs_absmax = 1.3624995346e+00`
- `eta_bulk_rhs_absmax / phi_chem_rhs_absmax = 1.5655146195e-01 mean, 1.7080876310e-01 max`
- `eta_Ldt_bulk_absmax / phi_Ldt_chem_absmax = 3.1034912063e-05 mean, 3.3861293126e-05 max`
- `eta_rhs_total_absmax / phi_rhs_total_absmax = 3.5366580305e-01 mean, 3.9872802763e-01 max`
- `max_deta_absmax = 1.3236165333e-03`
- `xB_min/max = [2.0903588893e-02, 5.4960119907e-02]` over the 100-step rerun
- `xBtot drift max = 1.4434123569e-04`, last `= 6.1865130560e-07`
- `eta_far_field_max = 3.3263892233e-04`

## Interpretation

1. **Does `dgbulk_deta` now use the same nondimensional chemical-potential scale as phi?**

Yes. The active raw branch now divides all chemical potentials by the same `mu_reference_scale` used by `mu_A_dimless / mu_B_dimless` in phi.

2. **Are eta RHS terms now internally code-unit consistent?**

Yes, for the active production path:

- chemical/bulk: converted to phi’s nondimensional chemical scale
- double-well: code-unit `gp_W_eta`
- gradient denominator: code-unit `gp_kappa_eta`
- mobility: code-unit `gp_L_eta`

3. **Is `eta_bulk_rhs_absmax / phi_chem_rhs_absmax` still raw-scale inflated?**

No. It dropped from `~1.97e3` to `~1.6e-1`.

4. **Must the previous `f_eta=1e-5` baseline be rescanned after the scale correction?**

Yes. The old baseline was stabilizing a raw-unit-inflated eta bulk term. After this correction, the same nominal `gp_L_eta` is far too conservative to use as a final physical baseline.

## Recommended next f_eta scan range

Do **not** continue using the old `f_eta=1e-5` baseline directly.

For the next clean scan, bracket upward from the corrected baseline using the **current** STO diffusion-limit reference and start in a much larger relative window, for example:

- `f_eta = 1e-7`
- `f_eta = 3e-7`
- `f_eta = 1e-6`
- `f_eta = 3e-6`
- `f_eta = 1e-5`

Reason:

- the current rerun used `gp_L_eta = 4.7495061735e-04`
- the same run reports `L_eta_diff_ref_code = 6.38333630e+03`
- so the actual ratio is only `7.44e-08` of the diffusion-limit reference

That means the old manually chosen mobility is now clearly in an **overly slow** regime after the scale fix, so the baseline should be rescanned upward rather than carried forward unchanged.
