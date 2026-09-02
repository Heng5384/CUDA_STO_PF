# Courant definition and audit

`GLOBAL_MAX_CFL` measures every actual FV face.  `ACTIVE_M0_CFL` and `ACTIVE_M3_CFL` use the smallest contiguous 99.9999% moment supports plus incident faces.  `LOWER_TAIL_ACTIVE_CFL` uses the M0 CDF band [1e-8, 1e-6].  `BOUNDARY_ACTIVE_CFL` is only active when that declared tail touches or reaches Rmin; the physical flux is nevertheless always executed.

First recorded step: global `2.870792113559984e+19`, M0 `0.7689546465160917`, M3 `0.5995405566018783`, lower-tail `1.1981215483739291`, boundary `0.0`.

Full per-step telemetry is in `courant_audit.csv` and `lower_tail_diagnostics.csv`.
