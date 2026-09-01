# Final acceptance

Top-level status: `FAIL_COHORT_EULERIAN_PHYSICS_MISMATCH`.

- Baseline: `PASS_RADIUS_GRID_BASELINE_REPRODUCTION`
- Smooth Eulerian authority: `PASS_EULERIAN_SMOOTH_POPULATION_AUTHORITY` at grid `3200`
- Cohort numerics: `PASS_DISCRETE_COHORT_NUMERICS`
- Cohort/Eulerian smooth crosscheck: `FAIL_COHORT_EULERIAN_PHYSICS_MISMATCH`
- Inherited CUDA A--E evidence: `PASS_INHERITED_CUDA_AE_EVIDENCE`
- Beta-only PF direction: `BLOCKED_PREREQUISITE_GATE`
- Local GP release: `LOCAL_GP_RELEASE_NOT_AUTHORIZED`

## Findings

- Legacy six-particle Eulerian P5 remains FAIL; the 2% threshold was not relaxed.
- Eulerian KWN authority is smooth-population grid 3200 only.
- Exact six-particle evolution uses no-bin event-aware cohorts and strict algebraic inventory closure.
- Frozen CUDA A--E evidence reuse is PASS_INHERITED_CUDA_AE_EVIDENCE with no CUDA rerun.
- Frozen Case A PF direction result is BLOCKED_PREREQUISITE_GATE with timescale NOT_RUN.
- The discrete event diagnosis identifies R10.5nm lower-bound numerical tail survival/leakage.

## Boundaries

CUDA A–E is frozen evidence reused without a rerun. PF source was not modified. No physical retuning, double counting, mass drift, GP release, GP→beta or online coupling was introduced. Historical authority remains unrecovered.

## Next action

Do not start local GP release; resolve the failed gate above in a separate, scoped task.
