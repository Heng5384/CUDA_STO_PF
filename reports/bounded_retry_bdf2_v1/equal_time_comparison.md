# Common-state equal-time comparison

All four runs start from byte-identical `Ctot`, `phi`, and `xB_alpha` fields with invalid BDF2 history, and therefore share the same BE startup. They end at code time 1.5625 (64.264977 s). The fine reference is active-manifold BDF2 at dt/32.

| Case | dt code | Steps | C increment error | phi increment error | h increment error | matrix profile error | interface error (dx) | reject fraction | reject wall | physical s/GPU h | Full gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dt4 | 0.00078125 | 2000 | 0.014% | 0.044% | 0.009% | 0.010% | 0.0001 | 0.750% | 9.857% | 1251.098 | FAIL |
| dt8 | 0.000390625 | 4000 | 0.009% | 0.011% | 0.006% | 0.008% | 0.0001 | 0.450% | 15.388% | 1151.036 | FAIL |
| dt16 | 0.0001953125 | 8000 | 0.000% | 0.001% | 0.000% | 0.000% | 0.0000 | 0.000% | 0.000% | 909.150 | PASS |
| fine_dt32 | 9.765625e-05 | 16000 | 0.000% | 0.000% | 0.000% | 0.000% | 0.0000 | 0.000% | 0.000% | 512.522 | PASS |

The errors use the preregistered increment-normalized definitions; the matrix profile is alpha-capacity weighted.
