# Final acceptance

Top-level status: `FAIL_LOWER_BOUNDARY_OPERATOR_PARITY`.

The old vague physics-mismatch label is retired for this task.  The status selector preserves the first failed gate rather than overwriting it with a later diagnostic.  The frozen 2% threshold is unchanged.

| Gate | Result |
|---|---|
| Baseline | `PASS_BASELINE_REPRODUCTION`; frozen contract `d0ff02973ab0f737043e1a40d4f69893a469cbfe2bc4cd22f9e6a410bd0b1333` |
| Canonical measure | `PASS_CANONICAL_INITIAL_MEASURE_IDENTITY`; M0/M3 authority errors `3.200e-16` / `1.518e-16` |
| Metrics / growth / matrix | `PASS_METRIC_IDENTITY` / `PASS_GROWTH_KERNEL_PARITY` / `PASS_MATRIX_CLOSURE_PARITY` |
| Interior moment rates | `PASS_SEMIDISCRETE_MOMENT_RATE_PARITY`; max error `6.649e-03` |
| Rmin flux parity | `FAIL_LOWER_BOUNDARY_OPERATOR_PARITY`; number/volume/mol-B errors `1.000e+00` / `1.000e+00` / `1.000e+00` |
| Constant-negative crossing | Eulerian returned `0.660919` of analytic `1` at C_R `0.034375` |
| -K/R crossing | Eulerian returned `0.660794` of analytic `1` at C_R `0.034363` |
| Literal CFL | current all-grid C_R,max `9.194740723e+06`; no 48 h passing raw-CFL policy is feasible |
| Authority / PF | `BLOCKED_LOWER_BOUNDARY_OPERATOR_PARITY` / `BLOCKED_LOWER_BOUNDARY_OPERATOR_PARITY`; beta-only comparison blocked |

No beta-only PF direction conclusion and no local GP release are authorized.  `PF_SOURCE_MODIFIED=False`, `CUDA_RERUN=False`, `PHYSICAL_RETUNING=False`.
