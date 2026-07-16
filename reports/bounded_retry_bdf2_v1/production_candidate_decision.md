# Production candidate decision

| Case | Reject fraction | Fallback fraction | Reject wall | Max run | Max depth | Persistent cell | p99 | Equal-time | physical s/GPU h | Contract |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|---|
| dt4 | 0.750% | 0.700% | 9.857% | 5 | 4 | True | 121 | PASS | 1251.098 | FAIL |
| dt8 | 0.450% | 0.450% | 15.388% | 2 | 2 | True | 69 | PASS | 1151.036 | FAIL |
| dt16 | 0.000% | 0.000% | 0.000% | 0 | 0 | False | 44 | PASS | 909.150 | PASS |

- **dt/4:** accuracy passes, but reject overhead, consecutive fallback, depth, and persistent-cell gates fail.
- **dt/8:** accuracy, frequency, depth and consecutive gates pass; reject wall overhead and persistent-cell gates fail.
- **dt/16:** all bounded-retry and equal-time gates pass with zero reject. It advances `64.265` physical seconds in `254.473` wall seconds (`909.150` physical s/GPU h), 1.77x the dt/32 reference throughput. This common-state 64.3 s window is sufficient to upgrade dt/16 from validation-only to the bounded-contract production baseline.

No evidence attributes the coarse-dt failures to BDF2 history. A new integrator is therefore not warranted here. The next numerical task is the bound-aware transport nonlinear solver if dt/4 or dt/8 throughput is needed. The optional 8 nm entry smoke was not run in this goal.

## Validation

- Python regression: `305/305 PASS`.
- Local/workstation bounded-retry host oracle: `PASS`.
- Local/workstation active-manifold host oracle: `PASS`.
- Workstation CUDA build: up to date and all four GPU runs exited 0.
- `compute-sanitizer`: unavailable on workstation; no sanitizer PASS is claimed.
- The unrelated hard-gate host target required `PF_T380_PARAMS`; it was not run because this goal explicitly prohibits T380.

`selected_production_dt_code=1.95312500000000011e-04`

`recommended_next_action=IMPROVE_BOUND_AWARE_TRANSPORT_NONLINEAR_SOLVER`

`final_status=PASS_BOUNDED_RETRY_DT16_PRODUCTION_QUALIFIED_DT4_DT8_SOLVER_BLOCKED`
