# Free-set second-order validation

Runtime matrix status: **PASS_BDF2_ORDER_AND_SHORT_WINDOW**. The active-manifold selector is exactly the original extrapolate on free cells.

| Pair | Observable | Coarse/fine error ratio |
|---|---|---:|
| dt_over_dt2 | Ctot_L2 | 3.974254 |
| dt_over_dt2 | Ctot_increment_L2 | 4.356053 |
| dt_over_dt2 | phi_L2 | 3.955922 |
| dt_over_dt2 | phi_increment_L2 | 3.377212 |
| dt_over_dt2 | hvolume_abs | 4.008264 |
| dt_over_dt2 | interface_abs | 4.041012 |
| dt2_over_dt4 | Ctot_L2 | 3.995040 |
| dt2_over_dt4 | Ctot_increment_L2 | 4.126602 |
| dt2_over_dt4 | phi_L2 | 3.991016 |
| dt2_over_dt4 | phi_increment_L2 | 4.430527 |
| dt2_over_dt4 | hvolume_abs | 4.011203 |
| dt2_over_dt4 | interface_abs | 4.020543 |

All registered Ctot, phi, h-volume, and interface ratios remain in [3,5].
