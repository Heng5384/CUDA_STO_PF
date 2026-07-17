# Baseline freeze

The accepted control is `LEGACY_CURRENT / V0 active-manifold IMEX-BDF2` at `dt=0.0001953125` code units
(`0.008033122153` s).  It completed 8000/8000
accepted steps with 0 hard rejects and 0 fallback macros.

The selected memory path uses feature mask `187`.  Its 8000-step endpoint is
bitwise identical for `Ctot`, `Ctot_nm1`, `phi`, `phi_nm1`, and `xB_alpha`; accepted iteration p99 remains
44.  Forced event rollback is bitwise.  Three-dimensional and restart controls are
listed in `optimization_ablation.csv`.

The worktree was dirty before this goal.  No reset, cleanup, commit, push, cluster execution, physics edit,
numerical-equation edit, or acceptance-gate edit was performed.
