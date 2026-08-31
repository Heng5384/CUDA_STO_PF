# PF contract audit for KWN–PF MVP v1

Status: `P0_CONTRACT_CONFLICT`.

This is a read-only audit. “Current local source” is not treated as a frozen
PF authority because the existing decision record has no contract hash and
does not bind the source to a qualifying binary, fixture, or result package.

## Controlling decision

`thermodynamic_contract_freeze_v1/07_frozen_thermodynamic_contract.yaml`
declares `BLOCKED_CONTRACT_CONFLICT`, has `contract_hash: null`, and records
source commit `1c08f9ee011b31e0cd4d82749e58a8f69ebd2204`. The later exact-fit
candidate is useful evidence, but the decision record explicitly says it is
not a deployed publication/runtime contract. Therefore no KWN `pf_contract`
adapter, PF/CUDA edit, PF initial condition, or PF execution is authorized by
this MVP.

## Composition and storage contract

| symbol / concept | PF code variable or expression | audited value / definition | unit | evidence path | conflict / status |
|---|---|---|---|---|---|
| pseudo-binary composition | `xB_alpha`, `xB` | `A=PbTe`, `B=Ag2Te`; `xAg=2*xB/(2+xB)`, `xB=2*xAg/(2-xAg)` | dimensionless | freeze YAML | conversion is frozen; APT need not sit exactly on the ideal tie line |
| beta composition | `v_B`, beta storage term | line-compound endpoint `xB_beta=1` | dimensionless | freeze YAML; `cuda_kernels.cu` two-phase storage comment | no conflict for endpoint |
| two-phase total B storage | `C=(1-h_beta)*xB_alpha+h_beta*v_B` | resolved matrix + beta storage | PF code concentration basis | `cuda_kernels.cu` two-phase storage kernel | current two-phase qualified path only |
| GP-mode total B storage | `xBtot_gp=h_alpha*xB_alpha+h_GP*xB_GP+h_beta` | legacy eta/GP storage relation | PF code concentration basis | `cuda_kernels.cu:3724–3740`; composition audit CSV | GP mode is not a current qualified KWN handoff state |
| GP composition | `gp_xB_fixed` | legacy pseudo-binary GP values appear around `0.35` / `0.3529411764705882` | dimensionless | `pf_params.h`; `main_cuda.cu`; legacy reports | not a validated GP thermodynamic contract or `x_AgI` mapping |

## Thermodynamic contract

| symbol / concept | PF code variable or expression | audited value / definition | unit | evidence path | conflict / status |
|---|---|---|---|---|---|
| alpha chemical potentials | `mu_PbTe_calphad`, `mu_Ag2Te_calphad`; `mu_A_dimless`, `mu_B_dimless` | freeze records `mu_A=G0_PbTe+RT ln(1-xB)+L*xB^2`, `mu_B=G0_Ag2Te+RT ln(xB)+L*(1-xB)^2` | J mol⁻¹ before nondimensionalization | `thermo_utils.h`; freeze YAML | P0: legacy/exact coefficient authority unresolved |
| alpha free energy | `g_alpha=(1-xB_alpha)*muA_alpha+xB_alpha*muB_alpha` | reconstructed from chemical potentials | code energy density | `cuda_kernels.cu:915–919` | runtime coefficient authority unresolved |
| beta / line compound | `h_beta`, `v_B=1` | beta has no independently fitted continuous KWN composition in v1 | dimensionless / code energy | `cuda_kernels.cu`; freeze YAML | KWN uses explicitly approximate curvature thermodynamics |
| interaction, legacy | `get_L_param(T)` | `L(T)=41212.9-18.05*T` | J mol⁻¹ | legacy/exact conflict audit; code-path audit CP001 | legacy historical/runtime evidence, not selected |
| interaction, exact candidate | `THERMO_DELTA_H_J_PER_MOL`, `THERMO_DELTA_S_J_PER_MOL_K` | `L(T)=41504.29119633958-18.469276826409214*T` | J mol⁻¹ | current `thermo_utils.h:262–364`; freeze YAML | candidate/local source visible, not hash-bound PF authority |
| 380 °C solvus, legacy | low-concentration root | `xB_eq=0.004664951821454188` at `T=653.15 K` | dimensionless | authority decision | not selected |
| 380 °C solvus, exact candidate | low-concentration root | `xB_eq=0.004649261005504821` at `T=653.15 K` | dimensionless | freeze YAML `reference_states` | candidate only |
| solvus equation | low-concentration root of `Delta_mu` | `RT ln(xB)+L(T)(1-xB)^2=0` | J mol⁻¹ | freeze YAML `driving_force_expression` | executable shared contract blocked |
| KWN curvature equilibrium | `DiluteEquilibriumAdapter.equilibrium_xb` | `x_eq(R)=x_eq,infinity exp[((2gamma/R+E_el)Vm)/(RT)]` | dimensionless | `src/kwn_mvp/thermo_adapter.py` | `APPROXIMATE_BETA_THERMO`, not PF-consistent |

## Material, transport, elastic, and discretization audit

| symbol / concept | PF code variable | audited value / definition | unit | evidence path | conflict / status |
|---|---|---|---|---|---|
| temperature | `temperature_K` | 380 °C = `653.15 K` | K | qualifying fixture and KWN configs | no temperature conflict in MVP |
| alpha / compound molar volume | `Vm_alpha_0`, `Vm_compound` | example/KWN assumption `4.1009e-05` for each; GP uses `Vm_g=Vm_alpha` only as explicit assumption | m³ mol⁻¹ | `physical_inputs.example.json`; KWN configs | active PF values still need contract binding; no implicit PF-code conversion |
| beta interface energy | `gamma_Jm2` | example `0.168`; Yu context lowest registries about `168–169 mJ m⁻²` | J m⁻² | example input; Yu constraint YAML | beta/PbTe plausibility only; not GP gamma |
| GP interface surrogate | `gp_gamma_alpha_gp` | example `0.05` | J m⁻² | `physical_inputs.example.json` | effective legacy GP parameter, not frozen GP thermodynamics |
| transport | `D_alpha`, `D_compound`, `D_mix` | code mixes `D_alpha`/`D_compound`; example has `D_compound/D_alpha=0.01` | m² s⁻¹ / ratio | `pf_params.h`; `thermo_utils.h:D_mix`; example input | no single authority-bound PF diffusivity for beta-only KWN |
| elastic contribution | `eps_iso_over_vB`, `eps_xx00…` | target principal eigenstrain `(0.046,-0.022,-0.017)`; example `eps_iso=0.00233` | dimensionless strain | `physical_inputs.example.json`; `cuda_kernels.cu` helpers | KWN only has declared mean-field penalty; it cannot reconstruct PF elasticity |
| grid, legacy/example | `dx`, `pf_dx` | `0.1 nm` | m | `physical_inputs.example.json` | conflicts with qualified fixture |
| grid, qualified profile | `pf_dx` | `1 nm` | m | `physical_override_T380_dx1nm_lambda4nm.json` | numerical-contract difference; not silently selected |
| interface width, example | `lambda_sm` | `0.6 nm` | m | `physical_inputs.example.json` | conflicts with fixture |
| interface width, smallest fixture | `lambda_sm_nm` | `4 nm` | nm | six-particle 96³ fixture JSON | validation-only fixture setting |

## Initialization, inventory, and extraction audit

| concept | code / artifact | audited value or path | status |
|---|---|---|---|
| qualified profile initialization | profile-library handoff | six fixed diffuse profiles with registered radii 8.0, 9.5, and 10.5 nm; `phi` and `delta_C_relaxation` hashes retained | preserve profiles; no sharp-sphere or scalar overwrite |
| smallest PF smoke fixture | `pf_mass_conserving_library_handoff_six_particle_96cube_v1` | 96³ periodic box, 380 °C, `dt_code=0.02`, `lambda_sm=4 nm`, mean `C_B_tot=0.03`; `validation_only` | only future smoke target; current gates prevent execution |
| GP release/birth controls | `gp_release_mode`, GP/beta debug fields in `pf_params.h` | technical paths exist | qualified two-phase handoff keeps GP, GP birth/release, external source, and GP-assisted beta modes off |
| PF global mass tolerance | `minimize_mass_tolerance_relative` | parameter exists, but no contract-hashed active value is established here | P0 item; KWN/handoff independently require `<=1e-10` |
| 246³ resolved beta PSD routes | `registered_particle_psd.csv` | `reports/pf_246cube_method1_production_authority_v1/authority/{A,B,C}/registered_particle_psd.csv` | locally retained tracks; no requested complete Broad/Narrow 400³ selection |
| 6 h / 12 h / 48 h PF context | existing report trees | `reports/pf_elastic_multi_particle_6h_fixture_v1/`, `reports/pf_6h_48h_coarsening_random_psd_v2/`, `reports/pf_246cube_three_seed_library_handoff_v1/` | evidence/fixtures only; no unified contract hash |

## Consequence for this MVP

The KWN backend may run only its separately labelled dilute/effective model.
The beta-only comparison is `P0_CONTRACT_CONFLICT`; no result is fabricated.
The one-way handoff package is allowed as a schema and mass audit, but non-zero
GP or sub-grid beta buckets remain package-only until PF gains qualified
persistent state variables. This is why the final status is
`PARTIAL_PF_STATE_NOT_CLOSED`, not a successful PF-coupling claim.
