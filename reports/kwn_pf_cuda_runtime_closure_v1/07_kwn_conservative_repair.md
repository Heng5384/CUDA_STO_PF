# 07 KWN conservative positivity repair

Repair method: `CONSERVATIVE_IMPLICIT_UPWIND_FACE_SOLVE` with `IMPLICIT_SHARED_FACE_SOLVE_WITH_FROZEN_START_STATE_VELOCITIES`. Face velocities are frozen at the start state and shared face fluxes are solved implicitly by an upwind tridiagonal M-matrix system. This is a numerical transport revision, not physical retuning.

The recorded numerical qualification execution used source commit `ebfaae4`. Later delivery-only changes do not alter that completed solver trajectory.

- Physical parameter retuning: `False`.
- Negative-bin clamp: `False`.
- P1 exact regression: `True`; P2 48 h completion: `True`; P3 no clipping: `True`; P4 conservation: `True`.
- P5 timestep convergence: `True`; P6 positivity utilization: `True`; P6 restart: `True`.
- Canonical 48 h residual: `0`; roundoff-zeroed bin count: `0`.

The conservative KWN trajectory is positive, ledger-closed, restart-identical and complete to 48 h, but the **radius/size-space grid** has not converged under the declared 200-versus-400-bin criterion. It is therefore not a qualified 48 h beta-only prediction.

At 48 h, the 200-versus-400 reference differences are N=`13.6975%`, Rmean=`4.3566%`, Rmean³=`15.9932%`, Sv=`5.3532%`, fβ=`0.1050%`, matrix xB=`0.5108%`. Every metric must be ≤2%; P5 radius-grid convergence is therefore `False` and overall status remains `FAIL_KWN_CONSERVATIVE_POSITIVITY`.

P1 audit: accepted steps `215`, minimum dt `0.8665787888498926` s, maximum CFL `0.35`, maximum utilization `0.2026004846811445`. Full 48 h audit: accepted steps `66651`, minimum dt `0.2174642339632555` s, median dt `1.855501183657518` s, rejected steps `0`.
