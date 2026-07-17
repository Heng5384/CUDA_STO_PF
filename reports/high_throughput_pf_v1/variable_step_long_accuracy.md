# Variable-step long-window accuracy

The variable trajectory ran 8000 accepted macros and was compared at matched
physical time with G9/dt4 and G12/dt32 trajectories from the same accepted history.

| Metric | Error | Gate | Pass |
|---|---:|---:|---|
| cumulative transfer | 1.179402e-03 | 3% | True |
| beta amount | 2.964863e-05 | 3% | True |
| interface trajectory | 4.043383e-04 dx | 0.5 dx | True |
| capacity-weighted matrix profile | 1.227147e-04 | 5% | True |
| far field | 4.462373e-05 | 2% | True |

Retry/fallback/overhead are 95.225000%, 0.000000%, and
43.384901%. Accepted dt mean/p50/p90/p99 are 1.286198846e-04,
9.765625000e-05, 9.765625000e-05, and 7.812500000e-04.
The dominant accepted dt bands are: 9.765625000e-05: 7608 steps (95.100% of steps, 72.206% of accepted code time), 7.812500000e-04: 343 steps (4.288% of steps, 26.043% of accepted code time), 1.953125000e-04: 11 steps (0.137% of steps, 0.209% of accepted code time), 2.441406250e-04: 3 steps (0.037% of steps, 0.071% of accepted code time), 3.051757812e-04: 3 steps (0.037% of steps, 0.089% of accepted code time), other 30 exact values: 32 steps (0.400%, 1.382% of accepted code time).
There are `37` accepted-dt increases, `17` reductions, and
`19` direction reversals. `7618` accepted macro indices
are associated with at least one rejected trial.

Status: `FAIL_VARIABLE_STEP_LONG_WINDOW_PRODUCTION_RETRY_GATE`.
