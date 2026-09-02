# Metric, growth-kernel, and matrix-closure parity

Metric status: `PASS_METRIC_IDENTITY`; growth status: `PASS_GROWTH_KERNEL_PARITY`; matrix closure status: `PASS_MATRIX_CLOSURE_PARITY`.

`Rmean_cubed=(M1/M0)^3` and `mean_R3=M3/M0` are now separate shared fields.  The growth CSV includes actual `KWNSolver.growth_rates` and `CohortSolver.growth_rates` paths at t=0, plus shared-radius probes at early/late matrix compositions.  The matrix CSV compares the Eulerian ledger and CohortSolver closures for four scaled canonical M3 states. Maximum growth-kernel relative difference is `0.000e+00`; maximum matrix-xB absolute difference is `8.674e-18` and residual is `0.000e+00`.
