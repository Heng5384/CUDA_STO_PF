# Recommended production architecture

## Decision

The qualified PF numerical path is:

```text
authoritative Ctot ledger
  -> ctot_jichen_imex_bdf2_active_manifold_v1
  -> fixed G9 transport gate
  -> fixed dt = 7.8125e-4 code = 0.032132488613 s
  -> memory mask 187
  -> full-audit cadence 100 with every-step authoritative hard gates
```

This path is accepted for local calibration and short PF transients. It is not
a practical 400-cube long-coarsening engine. The proposed long-time particle
handoff also is not yet production-qualified because its three-particle PF
overlap failed the radius and extinction-order contract.

Final decision class: **D**.

```text
recommended_next_action =
REFINE_PF_TO_PARTICLE_STATE_MAPPING_BEFORE_GP_CAMPAIGN
```

## What is qualified

- Memory mask 187 preserves the frozen fields, rollback, and restart contracts.
- Static allocation is 12.182601 GiB; measured peak is about 12.9 GiB, 83.44%
  of the workstation device.
- V3 physically scaled signed-defect gates pass on an unseen holdout.
- Fixed G9/dt4 passes short- and long-window QoI gates and the 400-cube P/M
  runtime gates with zero retry and zero fallback.
- Cadence 100 leaves mass, storage, bounds, phase KKT, cold transport residual,
  finite-state, clipping/projection, history, and applicable source/mechanics
  gates authoritative every step.
- PF-to-particle ledger roundtrip passes: global error `2.82816e-13`, particle
  beta partition error `3.97557e-13`, and unchanged particle count.

## What is not qualified

- Variable-step BDF2 is accurate on accepted trajectories but inefficient. Its
  8000-macro run had `95.225%` internal retry fraction and `43.3849%` rejected
  trial wall overhead. It remains default-off.
- The minimum useful 10x PF throughput gate is not reached. Fixed G9/dt4 gives
  `2.650718 physical s/GPUh` conservatively across P/M, or `7.00x` the frozen
  baseline and `2.772x` the clean strict control.
- A 50-physical-hour full-PF trajectory projects to about `67906 GPUh`, before
  GP release and elasticity costs.
- The particle oracle passes the two-particle overlap but fails the
  three-particle case: maximum radius error is `8.404%` and no registered
  extinction event occurs in the overlap window. No post-result fit was made.
- GP source, T380, elasticity-on 400-cube, and a formal long-time campaign were
  intentionally not run.

## Runtime roles

| Role | Selected path | Authority |
|---|---|---|
| strict numerical reference | G12/dt16 or G12/dt32 as registered by the comparison | validation only |
| PF production transient | fixed G9/dt4, mask 187, cadence 100 | accepted Ctot/phi state |
| variable-step path | default-off | diagnostic only |
| PF-to-particle conversion | offline conservative handoff | read-only snapshot conversion |
| long-time particle oracle | default-off | not production-qualified |
| GP-mediated source | not enabled | no authority in this goal |

## Conservation boundary

The PF snapshot ledger remains

```text
M_total = sum(Ctot) + M_GP
        = sum(q_alpha) + sum(h(phi) v_B) + M_GP.
```

The handoff partitions diffuse beta inventory exactly once. A future particle
model must preserve separate matrix, beta-particle, and GP inventories and may
not write back to PF or spend GP inventory without an explicit equal debit and
credit. The present roundtrip proves ledger mapping, not temporal closure.

## Required next closure

Before any GP campaign, establish a resolved PF ensemble in which the particle
model reproduces two- and three-particle radii, phase amount, and an actual
extinction event within the registered gates. Refinement must target state
mapping and particle dynamics provenance; it must not retune PF thermodynamics,
relax PF hard gates, or use a mass projection to hide disagreement.

Status: `CASE_D_MAPPING_REFINEMENT_REQUIRED`.
