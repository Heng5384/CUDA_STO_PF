# Performance hotspots

Nsight Systems availability: **AVAILABLE_AND_RUN_2025.1.3**. Nsight Compute availability:
**UNAVAILABLE_ON_WORKSTATION**.  Occupancy, register pressure, bandwidth and L2 claims are therefore made only
when directly present in `ncu_summary.csv`; unavailable metrics are not estimated.

The actual 400^3 representative curved benchmark used nonuniform `phi`, nonuniform chemical potential and
nonzero transport.  It completed 13 accepted steps with a measured process peak of
12788 MiB.  The first accepted step cost 93.691429 s; the final timed
summary and warmup policy are in `actual_3d_speed.csv`.

Existing event safety activated in 6 macros.  Dominant diagnostic:
`coupled_outer_max_iter_not_converged; BDF2 absolute mass-identity preflight also flags large-N roundoff`.  These recovered branches are included in the measured wall time and
were not hidden, retuned, or removed by the memory refactor.

Top profiler kernel: **compute_mu_x_from_xB_candidate_kernel(const double *, const double *, double *, double, double, double, double, double, double, double, double, const float *, const float *, const float *, double, int, int, double *)**.  The dominant engineering
cost remains repeated nonlinear transport/phase kernels and reductions; no arithmetic-order-changing fusion
or CUDA Graph was retained without a measured net benefit.
