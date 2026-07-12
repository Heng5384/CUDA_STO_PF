# CUDA_STO_PF - PF composition production rules

## Scope

This repository task is restricted to the PF-only composition solver.

The following components must remain OFF and untouched unless a later,
explicit task opens their gate:

- RSMD external source
- GP release
- GP growth/coarsening
- legacy GP eta storage
- external matrix reset
- direct beta inventory injection
- S3 reintegration

## Physics invariants

Use these definitions unless the actual source proves that the project uses a different symbol name:

`C_B_tot=(1-h)x_B_alpha+h*v_B`, `q_alpha=C_B_tot-h*v_B=(1-h)x_B_alpha`.

For `v_B=1`, every accepted state must satisfy `h<=C_B_tot<=1`,
`0<=q_alpha<=1-h`, and `0<=x_B_alpha<=1` on active matrix support.

The source-free conserved equation is `d_t C_B_tot=div(M_eff grad(mu))`.
A phase-only substep must keep local `C_B_tot` fixed and perform
`q_alpha_new=q_alpha_old-(h_new-h_old)*v_B`.

## Hard prohibitions

- Do not change physical free-energy or mobility parameters.
- Do not change seed profiles or seed files.
- Do not tune the solver to make beta grow.
- Do not use mass-losing clipping or a domain-wide physical mass projection.
- Do not silently suppress a failed cell or rate.
- Do not remove legacy L/X/Q modes.
- Do not claim runtime PASS without corresponding logs.
- Do not fabricate paths, line numbers, formulas, results, or hardware.

## Development rules

- Recover exact equations, operator order, FFT normalization, and history timing from source before changing equations.
- Preserve default legacy behavior until a new runtime flag is explicitly selected.
- Use double precision for conserved state, residuals, and diagnostics.
- Commit accepted state and history only after every stage passes.
- A rejected step restores every state, work buffer, history, counter, and diagnostic affecting later evolution.
- Every change requires a unit or operator-isolation test.
- Record each applied change with file, final line range, old/new formula, mass/bound semantics, and test coverage.
- Keep large runtime output out of Git.

## Required future modes

Primary: `ctot_fv_entropy_be`.

Comparator: `qalpha_fv_local_transaction`.

A tiny-grid explicit conservative implementation is a correctness oracle only unless separately accepted for production.

## Acceptance rules

A new mode may commit a step only when all applicable checks pass: finite state,
source-free mass error at most `1e-10`, roundoff local storage residual, no bound
violation, no clipping, zero physical projection, converged nonlinear residual,
and accepted energy/work audit.

T380 and S3 remain gated until T400 equal-time acceptance.
