# Active-manifold 502-macro event-window validation

## Decision

- dt/8: **PASS_PRODUCTION_EVENT_GATE**
- dt/16: **PASS_REFERENCE_EVENT_GATE**

| case | BDF2 | event BE | history BE | fallback | fallback-work estimate | Ctot error | phi error | h-volume error | profile error | interface error | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dt/8 | 500 | 1 | 1 | 0.199% | 2.645% | 1.780% | 0.057% | 0.021% | 2.073% | 4.050e-08 dx | 74.750 |
| dt/16 | 502 | 0 | 0 | 0.000% | 0.000% | 0.594% | 0.009% | 0.006% | 0.667% | 5.637e-09 dx | 29.487 |

The fallback-work estimate uses rejected plus excess accepted nonlinear-iteration
work because per-macro wall timestamps are unavailable. It is not inferred from
the slower nonlinear regime's total wall time. All accepted mass, KKT, and stable
endpoint energy/work rows pass; clipping and physical projection remain zero.
