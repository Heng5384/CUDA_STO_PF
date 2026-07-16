# Transactional harmlessness

The forced rejection smoke proves bitwise restoration of Ctot, phi, Y, and both BDF2 histories before a depth-2 BE recovery. Every natural common-state reject also records `rollback_bitwise=1`.

| Run | Rejects | Bitwise rollback | Rejected time unchanged | Accepted time strictly advances | Max mass error | Max phase KKT | Status |
|---|---:|---|---|---|---:|---:|---|
| dt4 | 15 | True | True | True | 9.877e-13 | 1.007e-10 | PASS |
| dt8 | 18 | True | True | True | 9.521e-13 | 1.003e-10 | PASS |
| dt16 | 0 | True | True | True | 9.948e-13 | 1.002e-10 | PASS |

Forced rollback smoke bitwise status: **True**. GP/source are disabled, so duplicate source/ledger actions are structurally absent. No accepted row reports clipping or physical projection.

Accepted-trajectory errors relative to dt/32 are: dt/4 Ctot `1.365e-04`, phi `4.419e-04`; dt/8 Ctot `9.327e-05`, phi `1.082e-04`. Thus retries do not create a material accepted-trajectory deviation.

`transactional_harmlessness_status=PASS`
