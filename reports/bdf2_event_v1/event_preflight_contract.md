# BDF2 event preflight contract

## Runtime selector

`BDF2_EVENT_PREFLIGHT_V1=1` is default-off and legal only with
`ctot_jichen_imex_bdf2_v1`.  It runs before a BDF2 transport solve and does not
modify `Ctot`, `phi`, either history field, or a physical parameter.

For every cell it evaluates

`C_anchor=(4 C_n-C_nm1)/3`, `phi_E=2 phi_n-phi_nm1`,
`q_n=C_n-h(phi_n)v_B`, and
`q_anchor_E=C_anchor-h(phi_E)v_B`.

The reason codes are versioned:

1. `PHI_EXTRAPOLATION_CONTEXT_INFEASIBLE`
2. `BDF2_ANCHOR_CAPACITY_INFEASIBLE`
3. `ACTIVE_SET_TRANSITION`
4. `ENERGY_CONTEXT_UNDEFINED`
5. `OTHER_VERSIONED_REASON`

A crossing of `q_alpha` or of its upper-capacity margin between the free and
active branches is an event.  Branch membership uses the solver's existing
`1e-12` active-set tolerance.  ULP motion inside an already-active branch is
not repeatedly subcycled.  This is a branch-topology predicate, not a
success-tuned residual threshold.  Phi endpoint checks use the pre-existing
64-double-ULP context envelope; no extrapolated value is clipped into a
physical state.

With `BDF2_EVENT_BE_SUBCYCLING_V1=1`, a preflight event selects
`event_action=BE_SUBCYCLE`.  Subcycling cannot be enabled without preflight.

## Implementation

- Pure classifier: `bdf2_event_utils.h:bdf2_event_classify_v1`
- CUDA preflight: `main_cuda.cu:ctot_bdf2_event_preflight_kernel`
- Unit test: `tests/test_bdf2_event_utils.cpp`

The two failures occupy the same `q_alpha=0` beta-side region.  The dt/8 cell
266 performs a material `FREE_TO_QALPHA_LOWER_CAPACITY` branch transition and
activates subcycling.  The dt/16 state is already lower-active within the
registered tolerance; it does not need repeated subcycling, and its former
failure is closed by the stable endpoint audit.
