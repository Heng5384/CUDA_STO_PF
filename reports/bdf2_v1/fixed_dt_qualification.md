
# Fixed-Step BDF2 Qualification and Selection

Automatic retry was disabled. Any rejection disqualifies that fixed dt.

| Candidate | dt code | Accepted | Reject | p99 transport iters | physical s/GPU-hour | Status |
|---|---:|---:|---:|---:|---:|---|
| dt/2 | 0.0015625 | 150/1000 | 1 | 188.460 | nan | FAIL_FIXED_DT_QUALIFICATION |
| dt/4 | 0.00078125 | 335/1000 | 1 | 268.980 | nan | FAIL_FIXED_DT_QUALIFICATION |
| dt/8 | 0.000390625 | 1000/1000 | 0 | 42.020 | 1034.703 | PASS_FIXED_DT_QUALIFICATION |
| dt/16 | 0.0001953125 | 1000/1000 | 0 | 24.000 | 896.206 | PASS_FIXED_DT_QUALIFICATION |

The selected dt is original dt/8: `3.90625000000000022e-04` code units or
`1.60662443064667065e-02` s per step. It completed 1000/1000 steps,
has one startup fallback only (`1.000e-03`), p99
transport/phase iterations `42.02` /
`7.00`, and throughput
`1034.703031` physical s/GPU-hour.

The equal-time dt/16 reference gives:

- Ctot increment relative L2 error: `1.241566e-05`;
- phi increment relative L2 error: `4.847192e-05`;
- h-volume increment relative error: `6.231126e-06`;
- interface error: `3.076691e-07` dx;
- capacity-weighted matrix-profile relative L1 error:
  `7.199667e-08`.

The raw unweighted matrix Linf is not used as a physical gate because it is
dominated by cells with vanishing alpha capacity; both raw and
capacity-weighted diagnostics remain reported.

Sanitizer: CUDA memcheck exercised BE startup plus two BDF2 elastic-on steps;
`ERROR SUMMARY: 0 errors`, `LEAK SUMMARY: 0 bytes`.

Legacy/P2: current and frozen binaries are bitwise equal for all three accepted
fields. Dedicated P1/P2/BDF2 tests, P2 deterministic reduction, adjoint, KKT,
state transaction, and 59-row storage oracle all pass.

`fixed_step_qualification_status=PASS_FIXED_DT_SELECTION`
