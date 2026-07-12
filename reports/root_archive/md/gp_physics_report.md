# GP Zone Physics Verification Report

## Status Summary

| GP Mechanism | Status | Evidence | Assessment |
|---|---:|---|---|
| Continuous GP field | present | `cuda_kernels.cu`: `compute_eta_rhs_kernel`; `phase_functions.h`: `phase_fractions_gp` | GP is represented by a dynamic `eta` field in `gp_zone` mode. |
| GP formation | partial | `gp_nuc_enabled` path in `main_cuda.cu`; eta initialization and PDE evolution | GP zones can form through event seeding and then evolve through eta dynamics. Pure spontaneous GP formation from a fully calibrated GP free energy is not established. |
| GP growth/coarsening | partial | `compute_eta_rhs_kernel`; `gp_kappa_eta`, `gp_L_eta`, `gp_W_eta` in `pf_params.h` | Eta has an Allen-Cahn-like PDE and gradient penalty, so growth/coarsening-like dynamics are possible. Quantitative GP thermodynamics remain surrogate/parameterized. |
| GP coupling to composition | present | `compute_Y_rhs_gp_kernel`; phase fractions from `phase_fractions_gp` | GP phase fraction affects composition storage and transport. |
| GP mobility/capacity contrast | present | `compute_J_alpha_gp_kernel` phase-weighted `M_eff`; `gp_M_GP`, `gp_M_beta` | GP changes local mobility/storage through interpolation. |
| Static or reservoir GP module | present | `enable_gp_assisted_beta_nucleation` path in `pf_params.h` and `main_cuda.cu` | A separate GP-assisted beta reservoir/debug path exists and is not the same as the continuous eta free-energy model. |

## Dynamic vs Static GP

The system contains **two GP concepts**:

1. A continuous dynamic GP field `eta`, coupled to PF transport and phase fractions.
2. A site/reservoir/event-driven GP-assisted beta nucleation path, which can inject beta nuclei and consume GP reservoir mass.

These two routes are not a single unified variational GP model.

## Verdict

The GP module is **partial**.

It is physically meaningful as a hybrid GP field plus event model, but it is not a fully self-contained GP-zone thermodynamic phase-field model. The strongest concern is duplicated GP representation: continuous `eta` dynamics and separate reservoir/event GP assistance can both influence beta nucleation.
