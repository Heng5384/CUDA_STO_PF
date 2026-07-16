# Fixed-step active-manifold qualification

| case | steps | BDF2 | event BE | history BE | fallback | internal rejects | max mass error | p99 iterations | physical s/GPU h | fixed-step gate / role |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dt4 | 2000 | 1976 | 14 | 10 | 0.700% | 15 | 9.877e-13 | 121.0 | 1250.210 | FAIL |
| dt8 | 1000 | 969 | 16 | 15 | 1.600% | 16 | 9.166e-13 | 102.0 | 300.859 | FAIL |
| dt16 | 2000 | 2000 | 0 | 0 | 0.000% | 0 | 9.948e-13 | 42.0 | 486.971 | PASS_VALIDATION_ONLY |

The dt/4 and dt/8 runs retain exact mass, accepted KKT/energy, bounds, zero
clipping, and zero physical projection, but fail the preregistered zero-reject
qualification. dt/8 also exceeds the 1% persistent fallback gate. dt/16 is the
only zero-reject fixed-step validation path; per the goal it remains a validation
integrator, not the final 3-D production integrator.

Active 2+2 restart equals continuous four-step execution bitwise for Ctot,
Ctot_nm1, phi, phi_nm1, and xB_alpha. The old BDF2 selector remains bitwise
identical to its frozen one-step reference. `compute-sanitizer` and
`cuda-memcheck` are not installed on the workstation, so sanitizer status is
`NOT_RUN_TOOL_UNAVAILABLE`, not an inferred pass.
