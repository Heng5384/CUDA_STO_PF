# PF smoke-test gate

Status: `NOT_RUN_P0_CONTRACT_CONFLICT`

No PF/CUDA executable was invoked.  The selected target is the smallest existing conditional fixture: `pf_mass_conserving_library_handoff_six_particle_96cube_v1` (96³, 380.0 °C, lambda=4.0 nm).  It is validation-only.

The run is blocked first by `BLOCKED_CONTRACT_CONFLICT`, and independently by the handoff state's `PARTIAL_PF_STATE_NOT_CLOSED` status.  Therefore the blank trajectory cells are intentionally not treated as zeroes or failed physical observations.

A future smoke test must preserve the existing diffuse profiles and `delta_C_relaxation`, start from the qualified fixture only, keep all prohibited birth/GP mechanisms off, and report continuity at 0, 6, and 48 h.
