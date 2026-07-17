# Long-window transport residual-gate validation

The strict `G12+dt/32` reference and every common-window candidate are evaluated at pre-registered 1x, 2x, 3x, and 4x windows. The accepted 1x checkpoint and BDF2 history are reused byte-for-byte; each long job continues for three further windows. Wall time and trajectory defect are combined across the first segment and continuation. The final window is 6.25 code time = 257.059909 s.

## Strict Reference

The `G12+dt/32` continuation passed unchanged hard gates with max mass error `9.948e-13`, transactional observed time `4.68749999999993339e+00`, and `3` event-safe fallback macros.

## Candidates

| Gate | dt | Hard | Retry/efficiency | Four-window accuracy | Local defect rate | Signed bias | Throughput (physical s/GPU h) | Status | Reason |
|---|---|---|---|---|---|---|---:|---|---|
| G12 | dt16 | True | False | True | False | False | 169.7 | FAIL | RETRY_EFFICIENCY_GATE;LOCAL_DEFECT_RATE_GROWTH_GATE;SIGNED_RESIDUAL_BIAS_GROWTH_GATE |
| G10 | dt16 | True | True | True | True | True | 1532.2 | PASS | NONE |
| G10 | dt8 | True | True | True | False | True | 2492.0 | FAIL | LOCAL_DEFECT_RATE_GROWTH_GATE |
| G10 | dt4 | True | False | True | False | True | 3845.1 | FAIL | RETRY_EFFICIENCY_GATE;LOCAL_DEFECT_RATE_GROWTH_GATE |
| G10 | dt2 | False | False | True | True | True | 1512.2 | FAIL | HARD_NUMERICAL_GATE;RETRY_EFFICIENCY_GATE |
| G9 | dt16 | True | True | True | True | True | 1907.5 | PASS | NONE |
| G9 | dt8 | True | True | True | False | True | 1047.0 | FAIL | LOCAL_DEFECT_RATE_GROWTH_GATE |
| G9 | dt4 | True | True | True | False | True | 3083.0 | FAIL | LOCAL_DEFECT_RATE_GROWTH_GATE |
| G9 | dt2 | True | False | True | False | True | 1180.7 | FAIL | RETRY_EFFICIENCY_GATE;LOCAL_DEFECT_RATE_GROWTH_GATE |
| G8 | dt16 | True | True | True | True | True | 1386.5 | PASS | NONE |
| G8 | dt8 | True | True | True | True | True | 1665.9 | PASS | NONE |
| G8 | dt4 | True | True | True | True | True | 1808.1 | PASS | NONE |
| G8 | dt2 | True | False | True | True | True | 4519.5 | FAIL | RETRY_EFFICIENCY_GATE |

Best fully passed long-window candidate: `G9+dt16`.
