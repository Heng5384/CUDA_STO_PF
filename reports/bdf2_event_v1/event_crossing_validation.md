# BDF2 event crossing validation

## Runtime result

Both frozen failures now cross transactionally without a hard reject, mass loss,
clipping, physical projection, or accepted-state nonfinite value. The stable V2
endpoint audit is finite on every evaluated BDF2 row. The event/rebuild checkpoint
test is bitwise identical for all five authoritative/history fields: **True**.

| case | accepted | BDF2 | event BE macros | history BE | fallback fraction | max depth | max mass error | wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dt/8 | 502 | 6 | 257 | 239 | 51.195% | 2 | 9.948e-13 | 277.605 |
| dt/16 | 502 | 438 | 32 | 32 | 6.375% | 2 | 9.379e-13 | 65.421 |

## Same-macro reference

The reference is eight Lie-BE eighth steps from the identical accepted state.

| case | Ctot increment error | phi increment error | h-volume error | interface error | matrix-profile increment error | direction |
|---|---:|---:|---:|---:|---:|---|
| dt/8 | 3.187% | 0.066% | 0.042% | 7.964e-08 dx | 3.707% | unchanged |
| dt/16 | 1.720% | 0.039% | 0.025% | 2.427e-08 dx | 1.958% | unchanged |

dt/16 passes the preregistered 2% Ctot/phi/h-volume and 3% matrix-profile
increment gates. dt/8 fails the Ctot and profile increment gates. More
importantly, dt/8 fallback is persistent (not a single known event): only
6 of 502 macros use BDF2. Its measured wall overhead
relative to the previously qualified event-free dt/8 baseline is
889.3%, far above the 5% production target.

## Gate decision

`STAGE_7_STATUS=FAIL_DT8_PERSISTENT_ACTIVE_SET_EVENT_FALLBACK_DOMINANCE`

The event transaction and energy audit are correct, but the required base dt/8
production trajectory is neither within the event increment-error envelope nor
within the fallback/overhead envelope. Stage 8 therefore remains gated.
