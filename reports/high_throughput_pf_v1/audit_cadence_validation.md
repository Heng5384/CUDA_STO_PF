# Production audit cadence validation

Authoritative hard gates remain every-step operations: mass, local storage and
bounds, phase KKT, cold transport residual, finite-state checks, clipping and
projection checks, history integrity, and every applicable source/mechanics
contract. The cadence selector controls only verbose/non-authoritative CSV rows
and full decompositions; it never changes an accepted field.

| Cadence | Wall time, 20 steps | Relative speed vs cadence 1 | Endpoint |
|---:|---:|---:|---|
| 1 | 1.00 s | 1.000 | reference |
| 10 | 0.94 s | 1.064 | bitwise equal |
| 50 | 0.92 s | 1.087 | bitwise equal |
| 100 | 0.91 s | 1.099 | bitwise equal |

All four 32-cube trajectories have identical hashes for `Ctot`, `Ctot_nm1`,
`phi`, `phi_nm1`, and `xB_alpha`; all have zero hard reject, retry, fallback,
clipping, and projection. Every emitted `CTOT_FULL_AUDIT_RESULT` states
`hard_gates_executed=1` and `authoritative_state_changed=0`.

An independent cadence-100 event smoke forced a rejected trial. Full audit was
executed on startup/history creation, the retry/fallback macro, history rebuild,
and final step. The event macro rollback remained bitwise and no illegal field
was allowed to wait for the next periodic audit.

Selected production cadence: `100`. Measured small-case audit speedup: `1.099x`.
This modest speedup is reported as measured and is not extrapolated to 400-cube.

Status: `PASS_AUDIT_CADENCE_100_AUTHORITATIVE_GATES_UNCHANGED`.
