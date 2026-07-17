# Low-memory transport baseline freeze

## Frozen implementation

- Review commit: `f440c0dcd4c02cd45d9079c35d3838ebfa9b37e2`.
- Physics: `pbte_ag2te_gp_coarse4_stoich_rd_v2`.
- Integrator: `ctot_jichen_imex_bdf2_active_manifold_v1`.
- Transport coordinate: `adaptive_logit_feasible_ctot_v1`.
- Accepted baseline: `dt_code=1.953125e-4`, `dt_physical=8.033122153233354e-3 s`, zero retry and zero fallback.

The source hashes in `baseline_manifest.json` are the V2 observer source set before the low-memory solver edits. The review worktree contains pre-existing uncommitted V2 observer work, so the review commit alone is not a complete binary provenance identifier. The baseline binary is frozen separately by SHA-256.

## Gate status

The strict reference residual gate is `1e-12`. No report currently demonstrates completion of the V2 normalized-defect long-window plus holdout qualification for a relaxed production gate. Therefore `1e-10` is permitted only as a solver-development diagnostic. It is not called a production gate in this work.

The V2 `G12/dt32` five-window holdout is running on the workstation. A clean low-memory binary build is intentionally deferred until that job exits so this goal does not disturb an unrelated GPU qualification run.

## Frozen acceptance invariants

The low-memory candidate does not change thermodynamics, mobility, BDF2 coefficients, phase PDAS/KKT equations, mass tolerance, storage bounds, or energy/work gates. It may not commit clipping or a domain-wide physical mass projection. GP, RSMD, source physics and S3 remain off.

Current status: `BASELINE_SOURCE_AND_BINARY_FROZEN_BUILD_PENDING`.
