
# Fixed-Step BDF2 History Contract

`history_contract_version=FIXED_STEP_BDF2_ACCEPTED_HISTORY_V1`

Accepted state owns `C_n`, `C_nm1`, `phi_n`, `phi_nm1`, `dt_n`, `dt_nm1`,
`history_valid`, integrator mode, accepted step, code time, and physical time.
Only the atomic commit at `main_cuda.cu:38383-38422` rotates history. A reject
restores current fields, prior fields, mechanics, counters, fallback state,
and diagnostics before retry.

Checkpoint files include `Ctot_nm1.raw`, `phi_nm1.raw`, and versioned metadata.
The loader rejects incomplete history and records the BE fallback reason.

Validation:

- direct active-history continuation: PASS;
- startup BE followed by BDF2: PASS;
- restart after startup and active BDF2 restart: PASS;
- four restart field/history comparisons bitwise equal: `True`;
- forced BDF2 reject vs clean half-dt control: five fields bitwise equal;
- rollback provenance equal: `True`.

`BDF2_history_status=PASS_ATOMIC_ACCEPTED_HISTORY`
