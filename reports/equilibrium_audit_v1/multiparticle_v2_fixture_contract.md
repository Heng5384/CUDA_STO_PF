# Multi-particle V2 fixture contract

This is the registered deterministic fixture for the validation-only
multi-particle V2 A/B/C comparison.  It is validation-only and must not be
described as an experimental particle-size distribution.

| field | value |
|---|---|
| grid | 96^3, `dx=1 nm`, periodic |
| temperature | 380 °C |
| particles | 3 |
| radii | 8, 10, 12 nm |
| centers | (20,20,20), (68,20,20), (20,68,20) nm |
| interface width | 4 nm (`tanh` half-width 2 nm) |
| profile A | order-independent maximum of the three analytic spherical tanh fields |
| profile B | full-model local minimization from A, then fresh raw-field dynamics |
| profile C | same B raw field, fixed-cell elasticity enabled |
| matrix root | `xB=0.004664951821454195` |
| GP/birth/release/nucleation | OFF |
| solver | `LEGACY_Y_SM_V1 + SM_EXPLICIT_CONTEXT_N_V1 + SM_TANGENT_N_V1 + JI_CHEN_CONSERVED_Y_ZERO_MODE_V1` |
| physical claim | deterministic qualification fixture only |

The minimum center distance is 48 nm.  The largest required exclusion is
`R_i+R_j+4 lambda = 34 nm`, so periodic-image separation passes.  The local
materializer is `scripts/materialize_multiparticle_profile_v1.py`.  The
corrected V2 raw-field hashes and all runtime output hashes are recorded in
`multi_particle_v2_results_v1.md` and `multi_particle_v2_restart_validation.md`.

## Current gate

The corrected 96^3 three-particle A/B/C runtime is complete in the isolated
profile-extension source.  All three cases have zero-mode PASS, deterministic
particle identities with no merge/split, and continuous/restart bytewise
equality.  This closes the validation-only multi-particle extension gate, but
does not promote the uncommitted extension to a clean production branch or
remove the separate planar/profile-equilibration blocker.
