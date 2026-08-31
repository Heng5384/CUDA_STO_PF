# Model limitations

## Authority and PF-state blockers

- `P0_CONTRACT_CONFLICT`: the PF thermodynamic freeze remains `BLOCKED_CONTRACT_CONFLICT`; this delivery does not choose legacy or exact-candidate coefficients.
- The requested full 12 h BROAD/REF and NARROW 400-cube resolved-beta PSDs are not retained locally, so beta-only PF comparison is not substituted with scalar moments.
- The current qualified PF state has no independent persistent GP inventory state and no sub-grid beta inventory state.  A non-zero such bucket is retained package-only, never added to the matrix.

## Effective KWN limits

- The prescribed GP source is a numerical coupling exercise, not GP nucleation prediction.
- Effective-CNT parameters are exploratory priors.  One soft-constraint hit among 64 retained sets does not identify GP composition, gamma, site density, attachment, diffusivity, or elastic penalty.
- The GP lower radius boundary is 1 nm because the stated observation scope is 1–3 nm.  Below-boundary dissolution is conservative but cannot resolve sub-nanometre cluster physics.
- KWN is mean-field and spherical-equivalent; it cannot reproduce spatial elastic competition, morphology, orientation, merging/splitting identity, or PF profile relaxation.

## What was intentionally not done

No PF/CUDA source was modified; no full 400³ campaign, new beta birth, GP-to-beta conversion, GP release, online coupling, dislocation physics, or physical retuning was introduced.
