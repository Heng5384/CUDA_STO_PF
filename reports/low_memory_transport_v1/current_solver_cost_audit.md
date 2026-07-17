# Current transport solver cost audit

## Measured source

The existing P1 CUDA-event profile contains 11 transport residual evaluations in one nonelastic profiled step. Median residual evaluation cost is `2.8398 ms`. The median components are `1.5859 ms` face flux plus divergence, `0.9610 ms` thermodynamics/mobility, `0.0981 ms` global reductions, `0.0959 ms` residual construction and `0.0914 ms` context reconstruction.

Face flux/divergence and thermodynamics/mobility therefore account for most of every failed trial. Reducing line-search trials and ending true plateaus early is the highest-confidence optimization direction.

## Per residual evaluation call graph

1. Reconstruct `xB_alpha`, `q_alpha`, `Y` and active status from the current conserved iterate.
2. Evaluate chemical potential and cell mobility.
3. Build three shared-face flux arrays and their conservative divergence.
4. Form the BE/BDF2 transport residual and capacity diagnostics.
5. Copy the small statistics packet and reduce `Linf`, `L2`, mass and `sum(divJ)`.

The feasible-coordinate preconditioned direction additionally uses one R2C FFT, one scalar spectral kernel, zero-mode removal, one C2R FFT, normalization and repeated tangent-set scalar reductions. Each line-search trial performs a complete residual evaluation.

## Evidence boundary

These timings predate the new low-memory build and are used only to select the optimization target. Fresh A-E profiles are pending because the workstation is occupied by the V2 holdout. No current candidate speedup is claimed yet.

Status: `SOURCE_COST_PATH_AUDITED_FRESH_ABLATION_PENDING`.
