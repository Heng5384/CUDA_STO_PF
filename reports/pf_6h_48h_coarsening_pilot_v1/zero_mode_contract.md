# Zero-mode and checkpoint contract

Selector: PF_CONSERVED_Y_ZERO_MODE_V1
Backend: HOST_NEWTON_BISECTION_V1
Provenance: SM_EXPLICIT_CONTEXT_N_V1 + SM_TANGENT_N_V1,
pf_composition_mode=legacy, pf_y_update_mode=lagged_rhs.

The frozen target mass is initialized from the exact t=0 raw fixture ledger.
Raw/VTK continuation is not accepted after initialization. Restart reads only a
versioned, checksummed checkpoint and rejects grid, dt, temperature, backend,
time-level, reaction-discretization, or parameter-fingerprint mismatches.
