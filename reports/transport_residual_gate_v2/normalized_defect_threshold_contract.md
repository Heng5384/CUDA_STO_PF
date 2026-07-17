# Normalized Defect Threshold Contract

This document freezes V2 before any new holdout is run. It does not modify or
replace the preregistered V1 `g_D <= 2` result.

## Hard gates

Each equal-time window and the full trajectory must satisfy:

- `eta_global <= 1e-4`.
- `eta_interface <= 1e-3`.
- `b_signed <= 1e-3`.
- `b_interface <= 1e-3`.

Each equal-time window must also satisfy:

- `eta_cell_max <= 1e-3`.

Signed bias fails the trend gate only when the window series contains three
consecutive strict increases, the final-window/first-window ratio is greater
than four, and the final-window absolute signed fraction is greater than
`1e-4`.

All existing long-window QoI, mass, bounds, KKT, energy/work, and
retry/fallback production contracts remain hard gates and are not relaxed.
The holdout QoI check explicitly includes the existing `0.02` Ctot- and
phi-increment relative-L2 limits in addition to transfer, h-volume, interface,
matrix-profile, far-field, and growth-direction checks.

After the unchanged V1 queue completes, the best candidate is selected among
rows that pass every non-peak hard/retry/QoI/signed-bias gate. A second candidate
is retained only when its physical-time throughput is at least 90% of the best.
Together with the strict reference this keeps the replay-plus-holdout budget at
no more than three main simulations.

## Strict-reference peak floor

For every strict `G12 + dt32` equal-time window, the observer records
`B_i=sum(abs(dt*R_i))` and the number of accepted updates `n`. The standard
floating-point summation bound is

\[
\gamma_n=\frac{n\epsilon_{64}}{1-n\epsilon_{64}},\qquad
E_{\rm round}=\max_i\gamma_nB_i.
\]

The independent local-peak diagnostic floor is

\[
r_{\rm peak,floor}=\max\left(
  p95(r_{\rm peak,ref}),
  \frac{100\max_kE_{\rm round}^{(k)}}{\Delta T_{\rm window}}
\right).
\]

Then

\[
g_{D,\rm floor}=
\frac{\max_kr_{\rm peak}^{(k)}}
{\max(r_{\rm peak}^{(common)},r_{\rm peak,floor})}.
\]

`g_D_floor` is diagnostic only. Its frozen classes are stable (`<=2`),
increased (`2..5`), warning (`5..10`), and strong warning (`>10`). A candidate
with all normalized, QoI, numerical, retry, and holdout gates passing is not a
physical-trajectory failure solely because of this localization diagnostic.

## Versioned outcomes

- Normalized failure: `FAIL_V2_MATERIAL_TRANSPORT_DEFECT`.
- Pass with `g_D_floor > 2`: `PASS_V2_WITH_LOCAL_PEAK_WARNING`.
- Pass with stable peak localization:
  `PASS_PHYSICALLY_NORMALIZED_DEFECT_GATE_V2`.

The V1 and V2 statuses must always be printed side by side.
