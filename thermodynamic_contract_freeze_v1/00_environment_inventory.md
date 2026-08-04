# THERMODYNAMIC_CONTRACT_FREEZE_V1 — environment inventory

Audit stage: Stage A (read-only)
Audit time: 2026-08-04 19:37--19:38 UTC
Current local branch: `CUDA_STO_PF-transport-v3-polarization-anisotropic`
Current local commit: `1c08f9ee011b31e0cd4d82749e58a8f69ebd2204`

## Environment discovery

| environment | identity | project path | source identity | scheduler / remote state | dirty state | status |
|---|---|---|---|---|---|---|
| LOCAL | `ZhihengdeMacBook-Air.local` | `/Users/heng/Documents/GitHub/CUDA_STO_PF` | branch `CUDA_STO_PF-transport-v3-polarization-anisotropic`, commit `1c08f9e` | local Git; `origin=git@github.com:Heng5384/CUDA_STO_PF.git` | heavily dirty; existing user changes preserved | FOUND |
| GITHUB | `github.com`, repo `Heng5384/CUDA_STO_PF` | remote only | default branch `main`; remote HEAD `d8e836566829eb3458641346cdaca6a2ddf3ed60` | Git remote reachable; no `gh` CLI installed; no exact-contract branch/tag found | remote refs unchanged | FOUND |
| WORKSTATION | SSH alias `workstation-tail`, host `fuxin` | `/home/zhiheng/PF/CUDA_STO_PF` | branch `main`, commit `a9f4b6ef91d6974f8395c6fe3169bf53c6422891` | SSH read-only access succeeded; no active `main_cuda` compute process | dirty with many deletions/untracked files; must not reset or overwrite | FOUND / DIRTY |
| CLUSTER source | SSH alias `uvip-cluster`, host `login1` | `/data/home/luozhiheng/CUDA_STO_PF` | branch `main`, commit `9db5fb4572523b126e49b7152c1272f745bdd2aa` | Slurm; SSH read-only access succeeded | dirty with modified runtime and untracked files | FOUND / DIRTY |
| CLUSTER campaign | `uvip-cluster` | `/data/home/luozhiheng/tmp/pf_400cube_psd_spatial_density_ladder_production_v1_20260803` | not a Git repository; campaign artifacts only | Slurm campaign directory; existing jobs observed | no source checkout identity; do not treat as deployable source | FOUND / NON-GIT |

## SSH identities discovered

- `workstation-tail` → `100.85.250.17`, user `zhiheng`.
- `cluster-direct` → `10.233.10.11`, user `luozhiheng`.
- `uvip-cluster` → `10.233.10.13`, user `luozhiheng`, ProxyJump `workstation-tail`.
- `workstation-direct` is configured but was not needed for this read-only audit.

## Git limitations and safety state

- GitHub has no branch named `CUDA_STO_PF-transport-v3-polarization-anisotropic` and no thermodynamic-contract tag.
- Git LFS is not installed in the local environment (`git: 'lfs' is not a git command`).
- No submodules are registered in the local checkout.
- The local worktree contains extensive pre-existing modifications and untracked reports/scripts. No reset, checkout, clean, pull, push, recompile, queue cancellation, or remote write was performed.
- Complete command/output evidence is in `terminal_output.txt`; local Git snapshots are in `.stageA_*` files.

## Calibration-path evidence added in audit revision 2 (2026-08-05)

The user-specified calibration workspace was found and audited read-only:

`/Users/heng/Desktop/1_Solubility_Calibration`

It contains a candidate-final exact four-point fit under
`outputs_exact_4pt_final/`, its reproducible implementation
`Exchange_expression_Fit_exact_4pt_final.py`, and the raw input
`Previous/Solubility_extract.csv`. Hashes are recorded in
`hashes/calibration_source_hashes.sha256`. This evidence changes the earlier
“exact source absent” finding to “exact candidate source present but publication
contract still blocked”; no remote or production modification was made.
