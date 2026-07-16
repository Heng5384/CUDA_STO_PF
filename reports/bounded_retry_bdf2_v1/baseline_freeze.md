# Bounded-retry baseline freeze

The old zero-reject evidence is preserved and remains a failure. This report does
not rewrite any file below `reports/bdf2_v1`, `reports/bdf2_event_v1`, or
`reports/active_manifold_bdf2_v1`.

| Baseline item | Frozen value |
|---|---|
| Old report tree SHA-256 | `726ce38d95cc9b68fd98c5c48e7c2ce7bcd7c43a517f008c113151bc4fc4568b` |
| Old `main_cuda.cu` SHA-256 | `815cafbba0912c55d3b8910ba66cca7a9329f656c01e279763b2c86a98e89b99` |
| Old workstation binary SHA-256 | `ae19503f2f3129c7fc34ef84e0603d3fe59bfa0bf1ff1c4c878eb8b33872b392` |
| Instrumented workstation binary SHA-256 | `3692e11ab8b05371b893a6ea4358880b10d774de433229ab51cec3c08efbf01a` |
| GPU / driver | `NVIDIA GeForce RTX 5080, 580.95.05` |
| CUDA compiler | `Build cuda_12.9.r12.9/compiler.36037853_0` |
| Host compiler | `c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0` |
| Grid | `512x1x1`, `dx=1 nm`, `lambda=4 nm` |
| Common code-time window | `1.5625` (`64.26497722586683 s`) |
| Physics/source/GP/elasticity | unchanged / OFF / OFF / OFF |

## Instrumented source hashes

| Asset | SHA-256 |
|---|---|
| `main_cuda.cu` | `db3dac3646c4beb3eb7cb989630a244942bc7de45ece81ffcb6b501343cd4e56` |
| `cuda_kernels.cu` | `a4d269067ab2de6059af1d48c333eda80aec43268c3f61036eab1a66e760b81a` |
| `cuda_kernels.h` | `3eb5cfa9ed35ddb533f43d5847ca270a0ac3a4ddb389770140e7fa7e7c425b53` |
| `pf_params.h` | `414792a90107526c8721245f71e66e9523ef382a13c58c68ae9cd7b1a1a3a1f2` |
| `active_manifold_bdf2_utils.h` | `e4fcce0513ee44fd462ebf711e247ffbc297cd73cefe3a26cf2f3039439489eb` |
| `bounded_retry_bdf2_utils.h` | `bb81c5996ddc232708643f7773258f24c6a78ba9e4b8379141a30f6a6b922e9c` |

## Common-state hashes

| Asset | SHA-256 |
|---|---|
| `Ctot_init.raw` | `c05457ef585dc40ae91aded43a8531cfd72213d8da7b82243105d63cd1c60524` |
| `phi_init.raw` | `3a69447f232dfefce4e0a2ef67c172654bcc1520de4e703c9fca4ecf25721acb` |
| `xB_init.raw` | `210314a5ff59199df91fc4c9bc9556795e520f41906ec62d10509252f5347b84` |
| `init_meta.json` | `bc8a7f5befb740612ae8579099f9393fa4781e44e880b27683194029fd1ce4a2` |

## Frozen controls

| Parameter | Value |
|---|---|
| `PF_RESEARCH_MODEL` | `pbte_ag2te_gp_coarse4_stoich_rd_v2` |
| `D_alpha` | `4.00000000000000057e+02` |
| `D_beta_for_calibration` | `0.00000000000000000e+00` |
| `D_compound` | `0.00000000000000000e+00` |
| `gamma_Jm2` | `1.68000000000000010e-01` |
| `lambda_sm_m` | `4.00000000000000025e-09` |
| `dx` | `1.00000000000000000e+00` |
| `temperature_C` | `4.00000000000000000e+02` |
| `ctot_nonlinear_max_iter` | `500` |
| `ctot_outer_max_iter` | `100` |
| `ctot_phase_linear_max_iter` | `500` |
| `ctot_step_max_retries` | `0` |
| `ctot_retry_shrink_factor` | `5.00000000000000000e-01` |
| `ctot_dt_min_ratio` | `9.76562500000000000e-04` |
| `ctot_transport_nonlinear_coordinate` | `adaptive_logit_feasible_ctot_v1` |
| `ctot_phase_semismooth_pdas_enabled` | `1` |
| `ctot_numerics_contract` | `ctot_jichen_imex_bdf2_v1` |
| `ctot_split_defect_policy` | `IMEX_BDF2_NO_POST_PHASE_POLISH` |
| `ctot_max_coupling_correctors` | `0` |
| `elastic_enabled` | `0` |
| `gp_growth_enabled` | `0` |
| `gp_initial_population_enabled` | `0` |
| `gp_literature_model_enabled` | `0` |
| `gp_nuc_enabled` | `0` |
| `gp_to_beta_enabled` | `0` |

The complete baseline parameter file is frozen by SHA-256
`5259ac5eb44876c381e84caf4bd68291ff415a94b33c9a56da45e6517530210f`. Its parsed content is reproduced below;
the equal-time runner changes only `dt`, the active-manifold selector, the two
existing event guards, and the default-off retry acceptance contract.

```text
D_alpha=4.00000000000000057e+02
D_beta_for_calibration=0.00000000000000000e+00
D_compound=0.00000000000000000e+00
E0_xx=0.00000000000000000e+00
E0_xy=0.00000000000000000e+00
E0_xz=0.00000000000000000e+00
E0_yy=0.00000000000000000e+00
E0_yz=0.00000000000000000e+00
E0_zz=0.00000000000000000e+00
GP_population_mode=OFF
L_phi=7.01173700729214144e-01
L_phi_calibration_mode=one_sided_diffusion_controlled
L_phi_code_value=7.01173700729214144e-01
L_phi_physical_value=3.38252293529426515e-11
PF_RESEARCH_MODEL=pbte_ag2te_gp_coarse4_stoich_rd_v2
PHASE_KINETICS_MODE=FINITE_LPHI_BE
S_11=2.14285714285714278e+02
S_12=1.19047619047619051e+01
S_13=1.19047619047619051e+01
S_14=0.00000000000000000e+00
S_15=0.00000000000000000e+00
S_16=0.00000000000000000e+00
S_22=1.40873015873015873e+02
S_23=8.53174603174603163e+01
S_24=-0.00000000000000000e+00
S_25=0.00000000000000000e+00
S_26=0.00000000000000000e+00
S_33=1.40873015873015873e+02
S_34=0.00000000000000000e+00
S_35=0.00000000000000000e+00
S_36=0.00000000000000000e+00
S_44=1.01190476190476190e+02
S_45=0.00000000000000000e+00
S_46=0.00000000000000000e+00
S_55=2.77777777777777786e+01
S_56=0.00000000000000000e+00
S_66=2.77777777777777786e+01
S_p_11=-8.62315016865079258e+01
S_p_12=7.28520259523809557e+01
S_p_13=6.84127650992063536e+01
S_p_14=4.31654932539682523e+00
S_p_15=-1.36337777777777780e-01
S_p_16=2.83887456349206335e+00
S_p_22=3.76984126984127101e+01
S_p_23=6.51305341269841165e+00
S_p_24=4.97322880952380952e+00
S_p_25=8.19816982142857142e+00
S_p_26=3.27075438492063508e+00
S_p_33=-4.95910785714286817e+00
S_p_34=5.62990827380952386e+00
S_p_35=9.24541535714285700e+00
S_p_36=3.70263422619047633e+00
S_p_44=-7.83538127380952290e+01
S_p_45=1.52213130952380959e+00
S_p_46=-4.55453878968253978e+00
S_p_55=6.90482859126984039e+00
S_p_56=1.00106347222222225e+00
S_p_66=-1.01126662698412773e+00
Vm_alpha_0=1.00000000000000000e+00
Vm_alpha_0_phys_m3mol=4.10089999999999999e-05
Vm_compound=1.00000000000000000e+00
W=1.00000000000000000e+00
beta_staged_conversion_enabled=0
calibration_script_hash=54649f59778da44cc3517ce8e696d566d18c218301e3ec3a30a78f081d211e5b
coarse_calibration_hash=e4292cef07159d0fae12d25ce13fe193f305b7d4a21e5a019b6f00187d2a464b
coarse_interface_mobility_a_M=0.00000000000000000e+00
coarse_interface_mobility_mode=off
coarse_model_name=pbte_ag2te_gp_coarse4_stoich_rd_v2
coarse_model_version=2
coarse_uncertainty_version=JI_CHEN_LONG_TIME_UNQUALIFIED_V1
composition_evolution_mode=ctot_mimetic_be
ctot_automatic_dt_growth=0
ctot_diagnostics_enabled=1
ctot_dt_min_ratio=9.76562500000000000e-04
ctot_elastic_validation_enabled=0
ctot_finite_interface_antitrapping_enabled=0
ctot_matrix_support_eps=1.00000000000000004e-10
ctot_nonlinear_max_iter=500
ctot_outer_max_iter=100
ctot_performance_profile_enabled=0
ctot_phase_linear_max_iter=500
ctot_phase_restart_solver_migration_allowed=0
ctot_phase_semismooth_pdas_enabled=1
ctot_retry_shrink_factor=5.00000000000000000e-01
ctot_step_max_retries=0
ctot_transport_nonlinear_coordinate=adaptive_logit_feasible_ctot_v1
dVm_alpha_dxB=0.00000000000000000e+00
diagnostic_rsmd_enabled=0
double_oracle_contract_hash=cc4cad8955684d45d34ca9db7d1b300dd82acc1fc7473ff52b23cd0f5cd3ae3e
dt=3.90625000000000022e-04
dx=1.00000000000000000e+00
dy=1.00000000000000000e+00
dz=1.00000000000000000e+00
elastic_enabled=0
elastic_shift_dimless=0.00000000000000000e+00
enable_gp_assisted_beta_nucleation=0
enable_legacy_gp_storage_coupling=0
enable_runtime_nucleus_library=0
eps_iso_over_vB=2.33000000000000004e-03
eps_xx00=4.59999999999999992e-02
eps_xy00=0.00000000000000000e+00
eps_xz00=0.00000000000000000e+00
eps_yy00=-2.19999999999999987e-02
eps_yz00=0.00000000000000000e+00
eps_zz00=-1.70000000000000012e-02
eta_accept=2.56992294984463391e-07
eta_floor_version=COARSE4_FP32_ETA_FLOOR_16_32_V1
fine_reference_hash=978f53126cf2d264d4c35886ff384853c806b8b15009b24e99e3d7ed26cb385c
finite_interface_calibration_max_points=1.20000000000000000e+01
finite_interface_calibration_min_points=1.00000000000000000e+01
finite_interface_production_min_points=1.20000000000000000e+01
finite_interface_resolution_test_override=0
finite_interface_violation_diagnostics_enabled=0
gamma_Jm2=1.68000000000000010e-01
gp_D_ratio=1.00000000000000002e-02
gp_L_eta_code=1.27736061704464410e+00
gp_L_eta_phys=6.16209874857207762e-11
gp_M_eta_ratio_to_crit=1.00000000000000002e-02
gp_W_eta_code=1.19047619047619047e+00
gp_W_eta_phys=6.00000000000000000e+08
gp_gamma_alpha_gp=5.00000000000000028e-02
gp_growth_enabled=0
gp_initial_population_enabled=0
gp_inventory_growth_enabled=0
gp_kappa_eta_code=1.48809523809523836e-01
gp_kappa_eta_phys=7.50000000000000124e-11
gp_l_eta_nm=1.00000000000000000e+00
gp_literature_model_enabled=0
gp_nuc_enabled=0
gp_radius_evolution_enabled=0
gp_reaction_nu_A=6.50000000000000022e-01
gp_reaction_nu_B=3.49999999999999978e-01
gp_stochastic_enabled=0
gp_to_beta_enabled=0
gp_xB_eq_alpha_for_eta=2.99999999999999989e-02
ic_phi_iface_w=2.00000000000000000e+00
ic_vf_init_phi=0.00000000000000000e+00
ic_vf_target_phi=4.00000000000000008e-02
init_case_tag=growth_N512_shift0_dt8_pre_event_freeze
kappa_phi=2.00000000000000000e+00
lambda_sm_m=4.00000000000000025e-09
mechanics_acceptance_mode=FP32_NORMALIZED_BACKWARD_ERROR_V1
mechanics_precision_mode=FP32_SPECTRAL
model_mode=two_phase
mu_reference_scale=2.06685360000000001e+04
pf_composition_mode=legacy
pf_params_schema_version=2
pf_y_update_mode=lagged_rhs
residual_normalization_version=DEALIASED_REAL_DIVSIGMA_OVER_KMAX_STRESS_V1
scheduled_nuc_enabled=0
t_real_unit=4.11295854245547687e+01
temperature_C=4.00000000000000000e+02
thermo_convex_extrapolation_enabled=1
thermodynamic_backend_hash=7d1fb79cd45b94f6ce50cd85f4e4b37f005838ff9c09fcd66818f8001bb84f99
v_A=0.00000000000000000e+00
v_B=1.00000000000000000e+00
y_update_mass_projection_enabled=0
zeta0_eta=8.67155078444999505e-01
zeta0_phi=1.00000000000000000e+00
zeta_eta=9.85318814972284599e+04
zeta_phi=8.84311661083619460e+05
ctot_outer_acceleration=OFF
ctot_preconditioner_a_ref=0.1
ctot_debug_transport_floor_audit=0
ctot_numerics_contract=ctot_jichen_imex_bdf2_v1
ctot_split_defect_policy=IMEX_BDF2_NO_POST_PHASE_POLISH
ctot_max_coupling_correctors=0
ctot_split_defect_skip_threshold=-1.00000000000000000e+00
ctot_split_defect_hard_cap=-1.00000000000000000e+00
ctot_split_defect_scale=1.00000000000000008e-30
```

`baseline_preserved=true`
