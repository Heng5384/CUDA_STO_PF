# PF Backbone Verification Report

## Scope

This report audits whether the current Ag2Te-PbTe codebase contains a phase-field backbone for composition, GP-zone, and beta-phase evolution.

## Status Summary

| Component | Status | Evidence | Assessment |
|---|---:|---|---|
| Composition field `x_B(x,t)` evolution | partial | `cuda_kernels.cu`: `compute_J_alpha_gp_kernel`, `compute_Y_rhs_kernel`, `compute_Y_rhs_gp_kernel`; `thermo_utils.h`: chemical potentials | Conserved composition transport exists through flux/divergence and storage variables. It is closer to a CH-like transport implementation than a fully documented textbook Cahn-Hilliard equation. |
| Chemical potential `mu` | present | `thermo_utils.h`; `cuda_kernels.h`: `launch_compute_mu_x_kernel`, `launch_compute_mu_C_gp_kernel` | CALPHAD-like chemical potentials are used by phi/transport kernels. |
| Mobility `M(x)` | present | `cuda_kernels.cu`: `M_alpha = D_alpha/G_alpha`, `M_eff = h_alpha*M_alpha + h_GP*gp_M_GP + h_beta*gp_M_beta` | Mobility is phase-weighted and composition/phase dependent through local thermodynamic factors. |
| Beta order parameter `phi_beta` | present | `cuda_kernels.cu`: `compute_phi_rhs_kernel`, `phi_semi_implicit_update_kernel` | Beta phase evolves by an Allen-Cahn-like PF equation with chemical, elastic, double-well, and gradient terms. |
| GP order parameter `phi_GP` / `eta` | present | `cuda_kernels.cu`: `compute_eta_rhs_kernel`; `phase_functions.h`: `phase_fractions_gp` | GP-zone field is dynamic in `gp_zone` mode and participates in phase fractions and storage. |
| Gradient energy term | partial | `pf_params.h`: `kappa_phi`, `gp_kappa_eta`; `cuda_kernels.cu`: semi-implicit updates | Gradient penalties are implemented for order parameters `phi` and `eta`. A clean explicit composition-gradient free-energy term `kappa_x |grad x|^2` is not established as a first-class backbone term. |
| Multi-phase interpolation | present | `phase_functions.h`: `phase_fractions_gp(phi, eta, ...)` | Alpha, GP, and beta fractions are constructed from `phi` and `eta`. |

## Architecture

```text
composition storage / xB
  -> chemical potential / thermodynamic factor
  -> flux J = M_eff grad(mu)
  -> conserved update through Y / xBtot

phi_beta
  -> chemical + elastic + double-well driving
  -> semi-implicit gradient update
  -> beta phase fraction h(phi)

eta_GP
  -> GP chemical/surrogate + elastic + double-well driving
  -> semi-implicit gradient update
  -> GP phase fraction (1 - h(phi)) h(eta)
```

## Verdict

The PF backbone is **partial**.

The code contains a real phase-field transport/evolution backbone for composition, beta order parameter, and GP order parameter. However, it is not a clean fully variational three-field PF model because composition-gradient energy is not clearly represented as an explicit `kappa_x |grad x|^2` free-energy term, and several nucleation/insertion mechanisms bypass the variational evolution.
