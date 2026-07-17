# Transport residual-gate baseline freeze

## Verdict

`baseline_preserved=true`. The gate study starts from commit `f440c0dcd4c02cd45d9079c35d3838ebfa9b37e2`, the
byte-identical common checkpoint, and the already-qualified active-manifold
IMEX-BDF2 contract. No physics, source, GP, elasticity, phase KKT, mass,
bounds, energy, retry, or iteration-budget setting is opened by this study.

| Item | Frozen value |
|---|---|
| Physics | `pbte_ag2te_gp_coarse4_stoich_rd_v2` |
| Integrator | `ctot_jichen_imex_bdf2_active_manifold_v1` |
| Retry contract | `ACTIVE_MANIFOLD_BDF2_BOUNDED_RETRY_PRODUCTION_V1` |
| Grid | `512 x 1 x 1`, `dx=1 nm`, `lambda=4 nm` |
| T | `400 C` |
| Common window | `1.5625` code time = `64.264977225866829 s` |
| Strict dt/16 | `1.95312500000000011e-04`, 8000 steps, 909.15 physical s/GPU h |
| Strict reference | `dt/32`, 16000 steps, gate `1e-12` |
| Historical report tree | `d80f2f86c08495c9ebe98550b996090eb950d9c12aa7fbc6b86e75b3a4721166` (1964 files) |
| Frozen workstation binary | `3692e11ab8b05371b893a6ea4358880b10d774de433229ab51cec3c08efbf01a` |
| Transactional observer binary | `3986440d157f29c5ab53387345f67bbedf7c264b1a5136b04461cec32a954b0b` |

## Common checkpoint hashes

- `Ctot_init.raw`: `c05457ef585dc40ae91aded43a8531cfd72213d8da7b82243105d63cd1c60524`
- `phi_init.raw`: `3a69447f232dfefce4e0a2ef67c172654bcc1520de4e703c9fca4ecf25721acb`
- `xB_init.raw`: `210314a5ff59199df91fc4c9bc9556795e520f41906ec62d10509252f5347b84`
- `init_meta.json`: `bc8a7f5befb740612ae8579099f9393fa4781e44e880b27683194029fd1ce4a2`
- `runtime.base.params`: `5259ac5eb44876c381e84caf4bd68291ff415a94b33c9a56da45e6517530210f`


## Source provenance distinction

The frozen workstation executable and its source hashes are retained exactly
as reported by the accepted bounded-retry evidence. The commit-frozen source
was independently compared with that staging source: `cuda_kernels.cu` and
`pf_params.h` differ bytewise only in whitespace, and whitespace-stripped
content is identical. The new binary adds only a default-off accepted-residual
observer. A 64-step diagnostics-off replay produced byte-identical `Ctot`,
`phi`, `xB_alpha`, `Ctot_nm1`, and `phi_nm1`; diagnostics-on produced the same
five hashes as well. A second 360-step on/off replay crossed real BE-subcycle
retries. The same five accepted/history fields remained byte-identical, while
the V2 observer committed 364 valid substeps to exactly 1.125 code time within
7.11e-15. Rolled-back partial subcycles do not enter `D_i`, `E_R`, or observer
time.

The complete machine-readable provenance is in `baseline_manifest.json`.
