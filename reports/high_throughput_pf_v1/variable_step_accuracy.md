# Variable-step accuracy qualification

## Evidence

All trajectories start from the same accepted step-5482 `Ctot`, `phi`, and BDF2 history.
The variable candidate uses G9 with `dt16 <= dt <= dt4`; the reference uses G12/dt32.
The variable interval is `1.48796508321118237e-01` code time. The strict endpoint mismatch is
`0.021248%` and the fixed G9/dt4 mismatch is
`0.241275%`.

| Metric | Variable error | Gate | Pass |
|---|---:|---:|---|
| Ctot increment L2 | 2.042491e-04 | 2% | True |
| phi increment L2 | 1.425605e-03 | 2% | True |
| h-volume increment | 2.045558e-04 | 2% | True |
| capacity-weighted matrix profile L2 (`q_alpha=Ctot-h`) | 2.467517e-06 | 3% | True |
| interface position | 1.439372e-05 dx | 0.25 dx | True |
| transfer | 2.045558e-04 | 2% | True |
| far field | 1.335112e-06 | 1% | True |

The 200-step run accepted all `200` macros with
`2` internal rejects and `0` fallback
macros. Mean accepted dt is `7.43982541605631457e-04` code units; p50/p90/p99 are
`7.81250000000000043e-04`, `7.81250000000000043e-04`, and
`7.81250000000000043e-04`. Maximum accepted normalized embedded error is
`8.72376247474686561e-01`. Maximum rejected-trial
error is `3.14371431595801321e+00`; rejected trials
were rolled back and are not part of the accepted trajectory.

The raw all-domain `xB_alpha` difference is
`9.800315e-02` and is diagnostic only. In beta support,
`1-h` approaches zero, so raw `xB_alpha` is not the conserved matrix profile and must not
be used as a production accuracy gate. The gated profile is `q_alpha=(1-h)xB_alpha`.

Status: `PASS_VARIABLE_STEP_SHORT_WINDOW_ACCURACY`.

This is a short-window one-dimensional accuracy qualification. It does not replace the
required 400-cube P/M throughput benchmark or establish a long-time GP model.

## Long-window result

The 8000-macro variable trajectory was compared at the same physical time with
1317 fixed G9/dt4 steps and 10537 strict G12/dt32 steps. Relative to strict,
cumulative transfer error is `1.179402e-3`, beta amount error `2.964863e-5`,
interface-position error `4.043383e-4 dx`, capacity-weighted matrix profile
error `1.227147e-4`, and far-field error `4.462373e-5`. Growth direction is
unchanged. Every registered long-window QoI gate therefore passes.

The composite variable-step qualification nevertheless fails because the same
trajectory has `95.225%` internal retry fraction and `43.3849%` retry-wall
overhead. This is an efficiency/controller failure, not a long-window physics
error. Detailed metrics are in `variable_step_long_accuracy_metrics.csv`.

Status: `FAIL_VARIABLE_STEP_LONG_WINDOW_PRODUCTION_RETRY_GATE`.
