# Thermodynamic Reference Consistency Audit

## Scope actually audited

Requested DFT/CE sources are missing from this checkout:

- `Ag_Int/`
- `ce_project/data/energies.csv`
- `ce_project/data/dft_training_structures/`
- `ce_project/outputs_2x2_only/`
- `ce_project/outputs_2x2_only/gp_free_energy_v11/`
- `ce_project/outputs_2x2_only/A09_ISIF2_vs_ISIF3/`
- `ce_project/outputs_2x2_only/gp_bulk_phase_minimal_analysis/`
- `memory_ledger_analysis.py`
- `界面能与化学势分析.txt`
- `Thermodynamic_Description_of_the_Ag-Pb-Te_Ternary_.pdf`

Therefore this audit can verify the current PF/CUDA/CALPHAD construction, and can flag the absence of a DFT-to-PF mapping, but it cannot validate numerical DFT formation energies, SOC status, ISIF status, or CE branch fits.

## Direct answers

**当前项目是否唯一确定了 `x_AgI -> x_B^PF` 映射？**

No. I found no local code or data path defining `x_AgI`, `y_I`, `N_Ag/N_PbTe_FU`, or a conversion into `xB_alpha`, `gp_xB_fixed`, or `xBtot_gp`. The visible solver uses `xB` exclusively as pseudo-binary `B = Ag2Te` growth-unit fraction. The requested DFT/CE directories that might contain `x_AgI` are absent.

**是否存在 `x_AgI = x_B` 的隐含假设？**

No explicit code assignment was found because `x_AgI` does not appear in the visible code. The risk is instead conceptual/documentary: reports and scripts use an effective `x_B^GP ~= 0.35` and mechanical-mixture GP reference, while the user brief says DFT structures are Ag-interstitial `Pb32Te32Ag_n`. Without a mapping file, treating those numbers as the same variable would be an implicit assumption outside the code.

**如果用 `x_B = x_AgI/2`，只是 Ag-count mapping 还是完整 thermodynamic mapping？**

It is only an Ag-count conversion under an extra assumption that two Ag atoms correspond to one `Ag2Te` B growth unit. It is not a thermodynamic mapping. The DFT branch is `PbTe host + n Ag -> PbTe:Ag_i`; the PF/CALPHAD branch is `PbTe-Ag2Te` pseudo-binary with SER-referenced PbTe and Ag2Te endpoints. A full mapping would also need Te bookkeeping/reservoir, reaction basis, reference offsets, entropy terms, coherent strain state, and units.

**当前 GP branch 能否直接接入 `x_B^alpha`？**

Only as the current surrogate PF model, not as a validated DFT GP branch. In `cuda_kernels.cu`, eta drive already reads `xB_alpha` and compares it to `gp_xB_fixed`; however the GP branch is either:

- `g_gp = (1-xB_gp) mu_A(xB_gp) + xB_gp mu_B(xB_gp) - gp_delta_g0`, or
- `mu_GP0 = (1-xB_GP) G_PbTe + xB_GP G_Ag2Te - gp_delta_g_stab`

Both are CALPHAD/pseudo-binary mechanical-mixture surrogates. They are not `f_GP(x_AgI,T)` from DFT/CE.

## Highest-risk findings

1. **CRITICAL: no independent DFT/CE `f_GP` branch is present in the audited workspace.** The GP eta thermodynamics uses pseudo-binary CALPHAD/mechanical-mixture references, not `Pb32Te32Ag_n` formation energies.

2. **HIGH: `gp_xB_fixed = 0.35` and `gp_reaction_nu_B = 0.35` are pseudo-binary B-unit quantities in code.** No evidence maps them from `x_AgI` or `y_I`.

3. **HIGH: negative or positive GP chemical drive can be created entirely inside the CALPHAD pseudo-binary surrogate.** The visible formula is `nu_A*mu_PbTe_calphad + nu_B*mu_Ag2Te_calphad - mu_GP0`. This does not prove a DFT Ag-interstitial branch is favorable or unfavorable.

4. **HIGH: DFT Ag-interstitial and CALPHAD Ag2Te references are incompatible without an explicit thermodynamic cycle.** `mu_Ag_metal` and `E(Pb32Te32)` are absent locally and cannot be subtracted from `G_Ag2Te_Solid`/`G_PbTe_Solid` by inspection.

5. **MEDIUM: diagnostic script `plot_step41D_gp_beta_driving_forces.py` duplicates CALPHAD formulas.** Its `GHSER_Pb` low-temperature branch appears inconsistent with `thermo_utils.h` and resembles the Te branch, so Step41D diagnostic values should not be treated as authoritative until reconciled with CUDA thermodynamics.

## Working verdict

The current visible project is internally organized around a PbTe-Ag2Te pseudo-binary PF model with an effective GP surrogate. I did not find direct code-level mixing of `x_AgI = x_B`, because the DFT/CE branch is absent and `x_AgI` is not referenced. But the thermodynamic reference risk is real: any previous negative GP chemical driving force obtained by directly comparing DFT Ag-interstitial formation energies to CALPHAD/PF `Ag2Te` pseudo-binary energies would be non-actionable unless an explicit mapping and reference-state conversion was made first.
