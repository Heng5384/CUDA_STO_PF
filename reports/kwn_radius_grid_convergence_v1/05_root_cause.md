# Root cause classification

Classifications: `IMPLICIT_UPWIND_NUMERICAL_DIFFUSION, DISCRETE_EVENT_SENSITIVITY_IDENTIFIED`.

- t=0 fixed-pivot M0/M3/beta inventory/total inventory pass the 1e-12 gate.
- Cell-integrated reconstruction differs from fixed-pivot M3 by at most 0.00179319 at t=0; it is diagnostic-only and does not alter dynamics.
- The 800 vs 1600 endpoint still exceeds the declared 2% gate under one identical output schedule; the remaining controlled change is radius-space resolution and its Rmin face-flux event timing.
- Passive tags receive the same frozen-velocity implicit M-matrix update as the aggregate and show grid-dependent lower-radius tails reaching Rmin; this is direct discrete evidence of first-order implicit-upwind numerical diffusion.
- The smooth PSD transport ladder converges while the exact six-particle fixture does not; the unresolved discrepancy is tied to discrete class transport/event timing rather than a global solver verdict.

## Final fixture P5 evidence

| Primary metric | 48 h relative error | P5 <=2% | Full-time maximum |
| --- | --- | --- | --- |
| N_m0_m3 | 7.684% | FAIL | 10.642% |
| Rmean_m | 2.507% | FAIL | 3.514% |
| Rmean3_m3 | 8.394% | FAIL | 12.046% |
| Sv_m_inv | 2.782% | FAIL | 3.903% |
| f_beta | 0.065% | PASS | 0.122% |
| matrix_xB | 0.314% | PASS | 0.560% |

The first registered final-pair P5 exceedance occurs at `6 h` for `N_m0_m3` with relative error `5.132%`.

## Controlled grid contract

| Grid | Rmin (m) | Rmax (m) | Edge SHA-256 | Timestep policy |
| --- | --- | --- | --- | --- |
| 1600 | 4.72403e-10 | 1e-07 | 14a92831638b41a67c21285030df911a914faeb424f3a995690d75020857eced | {"cfl_active_inventory_relative_threshold": 1e-06, "max_dt_s": 900.0, "min_dt_s": 1e-12, "positivity_safety": 0.999999, "rmax_outflow_relative_tolerance": 1e-10, "size_cfl": 0.35, "temperature_K": 653.15} |
| 3200 | 4.72403e-10 | 1e-07 | 05fcdca4a3dcaf3468cf5f3f4bb703b4ef09a09d85995c853226c70cd3e11164 | {"cfl_active_inventory_relative_threshold": 1e-06, "max_dt_s": 900.0, "min_dt_s": 1e-12, "positivity_safety": 0.999999, "rmax_outflow_relative_tolerance": 1e-10, "size_cfl": 0.35, "temperature_K": 653.15} |

Earliest observed tail event: `R8nm`, initial radius `8 nm`, grid `3200.0`, at accepted endpoint `0.00730797 h`. The route is the implicit-upwind lower face of bin 0 into the matrix ledger; it directly affects M0/N, while the fixed-pivot M3 inventory remains ledger-closed. A constant-start-rate extrapolation (explicitly not an event-time prediction) from that class's initial weighted dissolution rate gives `12.5608 h` to Rmin, versus the observed implicit-tail accepted endpoint above.

Across the final fixture pair, maximum Rmax outflow is `2.84157e-181` m⁻³ s⁻¹; maximum tag-vs-aggregate lower-number mismatch is `4.1943e+06` m⁻³ (relative tag closure `2.19615e-14`), and maximum ledger residual is `1.55406e-16`. This does not support `RADIUS_RANGE_TRUNCATION` or non-conservative lower-bound loss.

The t=0 M0/M3/beta-inventory/total-inventory invariants pass the 1e-12 projection gate; the cell-integrated versus fixed-pivot M3 difference is a recorded diagnostic reconstruction, not a dynamics change. Thus the observed final-pair discrepancy is not attributed to an initial M0/M3 projection mismatch or a post-processing substitution.
