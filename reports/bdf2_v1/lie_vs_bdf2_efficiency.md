
# Lie-BE versus IMEX-BDF2 Efficiency

The comparison uses the same RTX 5080, current binary, frozen T400 state, PF
physics, and registered 0.5% Ctot-increment error envelope.

| Method | dt code | dt physical (s) | Steps | Wall (s) | Physical s/GPU-hour | Ctot increment error |
|---|---:|---:|---:|---:|---:|---:|
| Lie-BE v2 | 2.44140625e-05 | 1.00414027e-03 | 1000 | 8.908601 | 405.776971 | 2.934172e-03 |
| IMEX-BDF2 v1 | 3.90625000e-04 | 1.60662443e-02 | 1000 | 55.898628 | 1034.703031 | 1.241566e-05 |

Both methods are within the same error budget; BDF2 is more accurate and
delivers `2.549930x` more
accepted physical time per GPU hour. Both report one transport and one phase
solve per step; elasticity is off in this timing comparison. The 512x1x1
resident-memory ledger is 0.11 MB for both. Final checkpoint I/O is included
in wall time but is not separately instrumented.

`BDF2_efficiency_status=PASS_BDF2_EFFICIENCY`
