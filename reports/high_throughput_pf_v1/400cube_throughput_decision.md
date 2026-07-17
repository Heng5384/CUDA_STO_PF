# Actual 400-cube throughput decision

All values below are measured on the same RTX 5080 with memory mask 187. Each
row uses three warmup accepted steps followed by five timed accepted steps.
Initialization, plan creation, and field I/O are excluded from steady-state time.

| Geometry | Candidate | wall s/step | mean dt code | physical s/GPUh | vs frozen | retry | fallback | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| P | strict_dt16 | 24.160 | 1.953125e-04 | 1.196979 | 3.16x | 0.000% | 0.000% | PASS_PRODUCTION_RUNTIME_GATES |
| P | fixed_G9_dt4 | 39.386 | 7.812500e-04 | 2.936992 | 7.75x | 0.000% | 0.000% | PASS_PRODUCTION_RUNTIME_GATES |
| P | variable_G9 | 17.708 | 9.765625e-05 | 0.816554 | 2.16x | 87.500% | 0.000% | FAIL_PRODUCTION_RUNTIME_GATES |
| M | strict_dt16 | 30.243 | 1.953125e-04 | 0.956231 | 2.52x | 0.000% | 0.000% | PASS_PRODUCTION_RUNTIME_GATES |
| M | fixed_G9_dt4 | 43.640 | 7.812500e-04 | 2.650718 | 7.00x | 0.000% | 0.000% | PASS_PRODUCTION_RUNTIME_GATES |
| M | variable_G9 | 21.281 | 9.765625e-05 | 0.679465 | 1.79x | 87.500% | 0.000% | FAIL_PRODUCTION_RUNTIME_GATES |

The conservative cross-geometry rate is the minimum of P and M. The fastest
qualified candidate is `fixed_G9_dt4`. The fastest measured
candidate is `fixed_G9_dt4` at `2.650718`
physical s/GPUh. The cost projection uses `fixed_G9_dt4` at
`2.650718` physical s/GPUh and is
`a qualified production projection`.
The corresponding measured speedup is `7.00x` relative to the frozen
0.378861 baseline.

The same final binary also removes the frozen baseline's false large-grid
retries, so the clean strict cross-geometry rate is `0.956231` physical s/GPUh.
Against that like-for-like strict control, fixed G9/dt4 provides `2.772x` more
accepted physical time per GPU hour. The larger `7.00x` number remains the
required comparison against the preregistered frozen baseline.

Variable G9 is not a production candidate: both geometries rejected seven of
eight first trials (`87.5%`) and spent `43.4-48.6%` of wall time on rejected
work. Accepted recovery steps were dt32, below the normal configured dt16
controller floor.

Status: `QUALIFIED_FULL_PF_THROUGHPUT_BELOW_10X_USEFUL_GATE`.
