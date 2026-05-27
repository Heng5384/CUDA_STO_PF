# Step38G dgbulk_deta Audit

## Scope
This step is diagnostic-only. No evolution equation, parameter, or update-order changes were applied.

## Code-path audit

- `phi` chemical RHS is computed in [cuda_kernels.cu:437](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:437) to [cuda_kernels.cu:450](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:450):
  - `muA = mu_A_dimless(xB, temperature_K, mu_reference_scale)`
  - `muB = mu_B_dimless(xB, temperature_K, mu_reference_scale)`
  - `delta_mu = mu0_compound - v_A * muA - v_B * muB - elastic_shift_dimless`
  - `partial_g_bulk = c_bulk * hp * (delta_mu - c_bulk * mu_total * volume_term)`
- `mu_A_dimless` / `mu_B_dimless` explicitly divide by `mu_reference_scale` in [thermo_utils.h:429](/Users/heng/Documents/GitHub/CUDA_STO_PF/thermo_utils.h:429) to [thermo_utils.h:437](/Users/heng/Documents/GitHub/CUDA_STO_PF/thermo_utils.h:437), so the `phi` chemical path is on the same nondimensional energy scale as the rest of the `phi` kernel.

- `eta` bulk term `dgbulk_deta` is computed in [cuda_kernels.cu:805](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:805) to [cuda_kernels.cu:829](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:829).
- In the current clean GP baseline, `gp_raw_reaction_drive_only = 1`, so the active branch is:
  - `muA_alpha_raw = mu_PbTe_calphad(...)`
  - `muB_alpha_raw = mu_Ag2Te_calphad(...)`
  - `minus_delta_mu_r_gp = nu_A * muA_alpha_raw + nu_B * muB_alpha_raw - gp_mu_reference_raw`
  - `dgbulk_deta = c_ref * (1 - h_phi) * hp_eta * (-minus_delta_mu_r_gp)`
- `gp_mu_reference_raw` is currently the mechanical-mixture host reference returned by [main_cuda.cu:5500](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:5500) to [main_cuda.cu:5504](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:5504), i.e. physical `J/mol`.

## Mathematical comparison

Current active `eta` bulk expression:

```text
dgbulk_deta
  = c_ref * (1 - h_phi) * h'(eta) * ( - [nu_A mu_A_raw + nu_B mu_B_raw - mu_GP0_raw] )
```

Current `phi` chemical expression:

```text
partial_g_bulk
  = c_bulk * h'(phi) * ( delta_mu_dimless - c_bulk * mu_total_dimless * volume_term )
delta_mu_dimless
  = mu0_compound_dimless - v_A mu_A_dimless - v_B mu_B_dimless - elastic_shift_dimless
```

Shared structural factors:
- both use `h_prime_of_phi(...)`
- both use a concentration-like prefactor (`c_ref` or `c_bulk`)
- both localize through interpolation factors

Critical scaling difference:
- `phi` uses `mu_A_dimless / mu_B_dimless`
- active `eta` raw branch uses `mu_PbTe_calphad / mu_Ag2Te_calphad` and `mu_GP0_raw`, i.e. physical molar free-energy scale

Therefore the current active `eta` bulk term is **not** on the same nondimensional energy scale as the `phi` chemical RHS.

## Interpolation convention

- `phi` uses `h_prime_of_phi(phi)` at [cuda_kernels.cu:425](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:425)
- `eta` uses `h_prime_of_phi(eta)` at [cuda_kernels.cu:802](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:802)

Conclusion:
- interpolation derivative convention is consistent
- this is **not** an `h'(q)` mismatch issue

## Baseline location diagnostic

Companion CSV:
- [eta_bulk_max_location_diagnostics.csv](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/data/step38g_eta_bulk_audit/eta_bulk_max_location_diagnostics.csv)

Maximum `|dgbulk_deta|` over the baseline `f_eta=1e-5` run occurs at:
- `step = 35`
- `idx = 466040`
- `(i, j, k) = (50, 54, 56)`
- `eta_before = 4.8469599177e-01`
- `phi_before = 4.9685715196e-01`
- `xB_before = 3.3819541249e-02`
- `Y_before = -3.3523118482e+00`
- `h(eta) = 4.703887...e-01` (from CSV field)
- `h'(eta) = 1.8742194101e+00`
- `eta_rhs_bulk = -2.9136558019e+03`
- `eta_rhs_double_well = 2.9995392983e-02`
- `eta_rhs_total = -2.9136258065e+03`
- `dxB = -2.5581160924e-05`
- `deta = 1.3237375865e-04`
- `region_class = interface`

Interpretation:
- the largest `dgbulk_deta` is spatially localized on the GP interface
- it is **not** a far-field/global-ordering maximum
- but its magnitude is overwhelmingly dominated by the bulk term, not double-well

## Numerical scale comparison

From the baseline attribution CSV:
- `eta_bulk_rhs_absmax ≈ 2.914e+03`
- `eta_bulk_rhs_absmean ≈ 1.075e+01`
- `phi_chem_rhs_absmax ≈ 1.478e+00`
- `phi_chem_rhs_absmean ≈ 2.138e-01`
- raw ratio `eta_bulk / phi_chem`:
  - absmax: `≈ 1.97e+03`
  - absmean: `≈ 5.03e+01`
- after multiplying by `L * dt`:
  - `eta_Ldt_bulk_absmax_over_phi_Ldt_chem_absmax ≈ 0.39` mean, `≈ 0.47` max

Interpretation:
- raw `eta` bulk RHS is about three orders of magnitude larger than `phi` chemical RHS
- the smaller chosen `gp_L_eta` partially compensates for this
- the observed “stable baseline” is therefore being achieved by a very small mobility in front of an oversized raw bulk term

## Conclusion

- `max |dgbulk_deta|` is **localized on the GP interface**, so this is not a far-field ordering artifact.
- `h'(eta)` uses the same convention as `h'(phi)`, so there is no interpolation inconsistency.
- However, the current active `eta` raw bulk term mixes physical molar free-energy quantities with a kernel that otherwise operates in code-unit thin-interface form.

Final classification:
- **[FLAG] unit/scaling bug in the active raw `dgbulk_deta` path**

This does **not** look like a spatial localization bug.
It looks like a scale-consistency bug: the active `eta` bulk term is too large in raw form, and current stability depends on choosing `gp_L_eta` small enough to compensate.
