# Thermodynamic source inventory

This inventory distinguishes the reproducible legacy runtime contract from the requested but undiscovered exact-fit evidence.

| source_id | file/path | local SHA-256 or state | formula / content | units | authority level | status / notes |
|---|---|---|---|---|---|---|
| LEGACY_RUNTIME_CUDA | `thermo_utils.h` | `5257598d8bc54f4d6b401a538eb187d044965fec014bad6de44f308f99b97cea` | `L(T)=41212.9-18.05T`; SGTE Pb/Ag/Te; regular-solution chemical potentials; Newton root | J/mol; T in K | AUTHORITATIVE_PRODUCTION_RUNTIME for existing legacy runs | present and runtime-reachable; no contract hash |
| LEGACY_RUNTIME_PYTHON | `Unit_Psedobinary.py` | `3273144cda4bba02912e82fd3007c1940482d55e54499876d656e3063710e1c1` | same legacy `L0_PseudoBinary`, `mu_PbTe`, `mu_Ag2Te`, `xAg2Te_eq_from_T` | J/mol; T in K | AUTHORITATIVE_PRODUCTION_RUNTIME for generated legacy params | present; duplicated numeric literals |
| LEGACY_GP_HOST | `main_cuda.cu` | source `d4809ca43e04034eeaa3df92df3dc93a0a196530ac34014aad850bedf3d20453` | `gp_literature_L_alpha0_J_mol=41212.9`, `alpha1=-18.05`; host GP regular-solution path | J/mol; T in K | LEGACY | current local source; no local binary |
| LEGACY_ANALYSIS | `analysis/compute_explicit_nucleation_rates.py` | present; hash to be recomputed after any clean source snapshot | computes `xB_eq` from legacy L | J/mol; T in K | GENERATED_DERIVATIVE / LEGACY | post-processing can silently diverge if not routed through contract |
| LEGACY_ANALYSIS | `tools/analysis/compute_schur_rc_predictions.py` | `36e948900fa103ae6eaba6555f61f8179cf9843ef2b7414802b48be970e3cb06` | duplicate `L_pseudobinary(T)` and root solver | J/mol; T in K | GENERATED_DERIVATIVE / LEGACY | used by CNT/Schur diagnostics |
| LEGACY_AUDIT | `reports/equilibrium_audit_v1/chemical_flat_equilibrium_root.csv` | `59662ea47699ba3ad2d076980e0f7c509832258598ce659ad5454bc1069a9a27` | legacy root at 380 °C: `xB=0.004664951821454188`, `xAg=0.004654096254055399` | dimensionless | FINAL_REPORT / LEGACY | internally reproducible but not exact fit |
| LEGACY_AUDIT | `reports/equilibrium_audit_v1/runtime_equation_audit.md` | `a6f7adc54045ed2b2b63cc483b0b63acffa2f755cd560d3c32c655ed728bca02` | documents legacy regular-solution equations | mixed | FINAL_REPORT / LEGACY | source audit, not exact-fit evidence |
| FIT_RECONSTRUCTION | `reports/equilibrium_audit_v1/thermodynamic_fit_reconstruction.md` | `8bf2b70db6c45ed2b9742c0a322ddfa200f19273f94d6cb47970ddc787be44b2` | explicitly states original real-unit barrier/fit sources are absent and values are not refit | mixed | FINAL_REPORT | evidence against claiming exact fit |
| FIT_RESIDUALS | `reports/equilibrium_audit_v1/thermodynamic_fit_residuals.csv` | `67c19f0ddbd63ce13ab6ecf07f6d2219e4cb3cfd57929071b278383b2262eb11` | `NOT_FIT`, residual `NA` | mixed | FINAL_REPORT | no four-point fit residuals |
| EXACT_REPORT | `exact_pseudobinary_fit_report.md` and `exact_fit_terminal_output.txt` | NOT FOUND locally, in Git objects, or remote branch listing | expected exact-fit authority | unknown | UNKNOWN | blocking absence |
| EXACT_DATA | `exact_fit_parameters.csv`, `exact_fit_pointwise_residuals.csv`, `exact_fit_sensitivity_summary.csv`, `exact_fit_downstream_impact.csv` | NOT FOUND | expected exact-fit outputs | unknown | UNKNOWN | blocking absence |
| EXACT_CANDIDATE | prompt values `DeltaH≈41504.291`, `DeltaS≈18.469277`, `xB≈0.0046492610` | no source hash | candidate only; not accepted as authority | J/mol, J/(mol K), dimensionless | UNKNOWN | cannot be frozen from prompt text |

## Confirmed legacy equations

```text
mu_A = G0_PbTe(T)  + R*T*ln(1-xB) + L(T)*xB^2
mu_B = G0_Ag2Te(T) + R*T*ln(xB)   + L(T)*(1-xB)^2
L(T) = 41212.9 - 18.05*T
```

The exact four-point solvus data, fit variable, residuals, covariance, and publication source are not present in the audited tree.

## Audit revision 2: calibration workspace evidence

The prior absence statement applies only to the Git repository tree. The
user-specified external calibration workspace now supplies the following
hash-pinned candidate exact sources:

| source_id | file/path | SHA-256 | formula / content | authority level | status / notes |
|---|---|---|---|---|---|
| EXACT_RAW_DATA | `/Users/heng/Desktop/1_Solubility_Calibration/Previous/Solubility_extract.csv` | `ce19c61dc2771a94905caa78e68fcd7caf0cd1a721306791572f908d4158d93d` | four PbTe-rich solvus points, Ag at.% and °C | AUTHORITATIVE_RAW_DATA candidate | raw local data; source attribution is indirect and digitization uncertainty is not embedded |
| EXACT_FIT_IMPLEMENTATION | `/Users/heng/Desktop/1_Solubility_Calibration/Exchange_expression_Fit_exact_4pt_final.py` | `92d929431ee4545c253686d6b6a47975797135f1ad16772f69d79594e031db6a` | exact xAg↔xB conversion, `Y=ln(xB)/(1-xB)^2`, linear fit | AUTHORITATIVE_EXACT_FIT_OUTPUT candidate | independently reproduced; external workspace, not Git-versioned with PF runtime |
| EXACT_FIT_REPORT | `/Users/heng/Desktop/1_Solubility_Calibration/outputs_exact_4pt_final/exact_4pt_final_fit_report.md` | `151ba40f0c8190afdb11fd849ff91702b80a7873963cc3a7f3dca88db03d5080` | ΔH=41504.3, ΔS=18.4693, R²=0.9778, root summary | AUTHORITATIVE_EXACT_FIT_OUTPUT candidate | final report says `PASS_FINAL_EXACT_4POINT_FIT`; rounded display values |
| EXACT_FIT_PARAMETERS | `/Users/heng/Desktop/1_Solubility_Calibration/outputs_exact_4pt_final/exact_4pt_final_parameters.csv` | `ffd5d4de8c01ee6b3d7b8699454e0b069ebb2b303aebdce026f97aabacd32da0` | rounded parameter table | AUTHORITATIVE_EXACT_FIT_OUTPUT candidate | no pointwise residual rows or covariance |
| EXACT_FIT_TERMINAL | `/Users/heng/Desktop/1_Solubility_Calibration/outputs_exact_4pt_final/exact_4pt_final_terminal_output.txt` | `56ffcdaf4643ccb451d00c106c0bdb6c840059a65f9d3058dfae5d08a5d5e8c4` | source path, four points, fit values, final status | AUTHORITATIVE_EXACT_FIT_OUTPUT candidate | confirms `PASS_FINAL_EXACT_4POINT_FIT` |
| CALIBRATION_CONFLICT_AUDIT | `/Users/heng/Desktop/1_Solubility_Calibration/paper_architecture_freeze_v1/02_numerical_conflict_audit.md` | `87aa549c91d706e1bfed44ddda7c356a580afc359c09bf54c7429f8e79898f64` | explicitly records exact/runtime conflict and blocking action | CONFLICTED | says one publication coefficient contract is not yet selected |

Independent read-only reproduction gives `ΔH=41504.29119633958 J/mol`,
`ΔS=18.469276826409214 J/(mol K)`, and
`xB_eq(653.15 K)=0.004649261005504821`. The calibration workspace itself
still identifies the direct-composition residual CSV, complete bibliographic
source traceability and downstream PF impact as unresolved blockers. Therefore
these exact sources are candidate authority, not yet a deployed contract.

After the user's explicit local-update instruction, the current local branch
source `thermo_utils.h` and the listed analysis/runtime literals were patched to
the exact candidate. The local source remains dirty and unbuilt; workstation,
cluster, existing binaries, checkpoints and old result provenance remain
legacy/unknown and were not changed.
