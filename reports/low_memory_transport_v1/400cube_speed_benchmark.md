# 400-cube speed benchmark

No 400-cube speed run was attempted. The source-exact PF-only allocation ledger
is 19.580 GiB at 400^3, already above the RTX 5080 85% admission ceiling of
13.533 GiB before CUDA context and cuFFT workspace. In addition, no low-memory
solver passed the residual/fallback prerequisites. Consequently the result is
`400CUBE_SPEED_NOT_MEASURED_MEMORY_AND_SOLVER_BLOCKED`, not an extrapolated or
actual throughput claim.
