# Variable-step history and restart contract

The accepted checkpoint stores `C_n`, `C_{n-1}`, `phi_n`, `phi_{n-1}`, `dt_n`, `dt_{n-1}`,
`history_valid`, accepted step/time, fallback/event state, controller-valid state, previous
normalized error, and proposed next `dt`.

Only accepted steps rotate history and commit the PI controller. Any failed trial restores the
complete transaction before retry. A BE event/history rebuild invalidates the BDF2 estimator until
a legal two-level history exists again.

The restart gate permits one narrow migration into the default-off variable mode: a checkpoint from
the fixed active-manifold BDF2 family may be used only when the PF research model, split policy,
corrector count, energy contract, state dimensions, time provenance, and both raw history fields
match. No authoritative field is transformed. Controller state is initialized rather than
invented. All other physics/numerics contract mismatches remain fatal.

The workstation smoke migrated checkpoint step 5482 and emitted both
`CTOT_ACTIVE_MANIFOLD_CONTRACT_MIGRATION` and `CTOT_VARIABLE_BDF2_HISTORY_MIGRATION`. Its final
checkpoint records `VARIABLE_STEP_IMEX_BDF2_ACTIVE_MANIFOLD_V1`,
`VARIABLE_STEP_BDF2_ACCEPTED_HISTORY_AND_CONTROLLER_V1`, valid history, valid controller,
`dt_n=dt_{n-1}=7.8125e-4`, and `next_dt=7.8125e-4`.
