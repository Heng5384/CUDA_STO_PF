# Literal face-Courant accuracy ladder

Status: `FAIL_EULERIAN_TIME_ACCURACY`.  The current 0.1 h diagnostic starts at literal all-grid `C_R,max=9.194740723e+06`, while its legacy active-cell CFL is `0.35`.

A literal C_R<=1 policy begins at dt `2.099338273e-07` s and projects to `8.231165e+11` accepted steps and `8.756902e+01` serial years for 48 h at the measured micro-probe cost.  The requested 48 h ladder was therefore not falsely represented as complete; the recorded policies are reproducible micro-probes demonstrating that the cap is actually applied.
