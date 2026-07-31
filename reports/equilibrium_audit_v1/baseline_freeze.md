# T380 workstation equilibrium audit — baseline freeze

- source commit: `6b69895af2d1b86b99c57c5479ff767349c61efe` (detached HEAD; no commit/push performed)
- working-tree status at freeze:

```text
## HEAD (no branch)
 M main_cuda.cu
?? jobs/submit_pf_6h48h_preflight_v1.sbatch
?? jobs/submit_pf_6h_48h_pilot_v1.sbatch
?? reports/conditional_microstructure_path_v1/
?? reports/energy_minimization_T350_T450_cluster_20260727/
?? reports/fixed_dt_low_s_production_v1/
?? reports/pf_6h_48h_coarsening_pilot_v1/
?? reports/pf_ctot_qalpha_validation/
?? reports/pf_only_x_q_validation/
?? reports/workstation_s_nuc_eta_fit_dt0p02_v1/
?? scripts/analyze_pf_zero_mode_checkpoints.py
?? scripts/generate_equilibrium_audit_v1.py
?? scripts/materialize_pf_6h_48h_fixture.py
```

- temperature: 380 °C = 653.15 K
- audit mode: beta PF only; GP, GP birth, GP release, source and beta nucleation OFF
- frozen parameter candidate: `Results/interface_width_audit/pf_strict_dualdx_final.params`
- parameter SHA-256: `41b23b10ad6554b124eae48046d99fd40c86168a95cb4b6c07ea09b60dee3a83`
- no production parameter file was modified.

## Source hashes

| file | SHA-256 |
|---|---|
| `main_cuda.cu` | `1949c84a4d7026ac7b30a2fcccf1b4a0a0f8f0f8fb41f0dc67e0ae0917ee68e7` |
| `cuda_kernels.cu` | `f27db426148e25d6d9b221ad24e5013c522d8108c08f86db036807471faee0c6` |
| `cuda_common.cu` | `1d10b910b8557f997bf2ac2db4f02931372f8f6f4c3b42b788a5f3901e7e2d3d` |
| `thermo_utils.h` | `5257598d8bc54f4d6b401a538eb187d044965fec014bad6de44f308f99b97cea` |
| `phase_functions.h` | `1de58a0c93208688d767bec1a3fa8b70b9bf0ec25f7a8d48ac5de0d3f651ac52` |
| `pf_params.h` | `752db9e072bb98a4c0c233b7fef4b40501b10beb417d074c318a9f4083a0889a` |
| `pf_zero_mode_checkpoint.cpp` | `559eecf734041782c9fc9e36406358c75ad5e4ae2760d174d9cf10616840e453` |
| `pf_zero_mode_checkpoint.h` | `9d91278bd9b333a41a6d418b4ff3b46eefb2c420875cad9671d0864296e35008` |
| `Unit_Psedobinary.py` | `b666627e906544bdaad1f233065f541fcdfe88571b1173498635cc87770c1a43` |
| `physical_inputs.example.json` | `390b822cf04f40ad1e79a48662414cf100be33113411dc8e356171239913550a` |
| `Makefile` | `b571c627d75e5c962ef31e7efa01e51f20100a06622d8549ee6f339915d7f204` |

The CUDA binary is not part of this repository freeze; workstation binary/hash is recorded only after the isolated audit build.

## Isolated workstation audit build

- workstation: `fuxin` / RTX 5080 (16,303 MiB)
- isolated root: `/home/zhiheng/tmp/codex_equilibrium_audit_v1_20260729`
- binary SHA-256: `d5e87d8eb2d3a839934454031d6350f96d8b8865ee45cccdb58063ae27e8d999`
- audit parameter fixture SHA-256: `e69620fab71521a9b28659ea05660e7545bb5dae24fd21a9e9074f241c78c3aa`
- CUDA toolchain: `/usr/local/cuda-12.9`, `sm_120`
- all workstation outputs are in the isolated root; no existing production output root was overwritten.

## Authoritative branch build

The authoritative qualification source is the branch `codex/pf-zero-mode-restart-provenance-v1`, which is checked out in the sibling worktree `/Users/heng/Documents/GitHub/CUDA_STO_PF-jichen-real-s-eta-production-runtime-v1` at the same commit. The current audit worktree cannot switch to that branch because Git already has it checked out in that sibling worktree; this is a Git worktree lock, not a source mismatch. The branch source was therefore synchronized separately to:

`/home/zhiheng/tmp/codex_equilibrium_audit_v1_branch_20260729`

Authoritative branch hashes:

- `main_cuda.cu`: `e25de4dc23a5cbf961b0594ae40e466727bfe43c6118d24ce0119182f7457237`
- `cuda_kernels.cu`: `f27db426148e25d6d9b221ad24e5013c522d8108c08f86db036807471faee0c6`
- `thermo_utils.h`: `5257598d8bc54f4d6b401a538eb187d044965fec014bad6de44f308f99b97cea`
- compiled branch binary: `11d073a272a0b7fa668e40037bc1f96b8389909f95c74d6ad9a5db11236e01af`

The earlier short runs under `/home/zhiheng/tmp/codex_equilibrium_audit_v1_20260729` used the dirty current-worktree `main_cuda.cu` and are retained only as non-authoritative diagnostics. Branch qualification evidence begins with the separate `branch_R8_*` runs.
