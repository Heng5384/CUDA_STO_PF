# Variable-step retry and restart validation

The 200-step T400 frozen-history accuracy trajectory accepted all 200 macros.
It had two internal rejected trials (`1.0%`), zero fallback macros, and no macro
hard reject. Rejected-trial wall overhead was `0.4133%`, below the 5% gate.
Maximum accepted embedded error was `0.872376`; the rejected maximum was
`3.143714` and was atomically rolled back. Mean accepted dt was
`7.439825416e-4`, with p50/p90/p99 all equal to dt4 (`7.8125e-4`).

The short-window QoI comparison against G12/dt32 passes all registered gates,
including the conserved capacity-weighted matrix profile. Mass, storage,
bounds, KKT, cold residual, energy/work, finite-state, clipping, and projection
checks remained hard gates.

Restart was tested independently from the same accepted step-5482 state. A
continuous 40-step variable trajectory and a 20-step plus 20-step restarted
trajectory are bitwise identical for `Ctot`, `Ctot_nm1`, `phi`, `phi_nm1`, and
`xB_alpha`. Final `dt_n`, `dt_nm1`, controller previous error, controller next
dt, accepted time, and history-valid flags are exactly equal.

Status: `PASS_VARIABLE_BDF2_RETRY_AND_RESTART_CONTRACT`.

## Production-scale retry result

Transaction safety does not imply production efficiency. In the 400-cube P
and M eight-macro benchmarks, each variable run rejected seven internal trials:
retry fraction was `87.5%`, with retry-wall fractions `48.576%` and `43.405%`.
Both cases then accepted `dt32`, below the configured normal `dt16` floor.

The independent 8000-macro long window is stronger evidence: `7618` internal
trials were rejected (`95.225%`), zero fallback macros occurred, and rejected
trials consumed `43.3849%` of wall time. The accepted mean dt was
`1.286198846e-4`; p50 and p90 were `9.765625e-5`, while p99 was
`7.8125e-4`. Of 8000 accepted macros, 7608 used dt32.

Final status: `FAIL_VARIABLE_BDF2_PRODUCTION_RETRY_AND_OVERHEAD_GATES`.
