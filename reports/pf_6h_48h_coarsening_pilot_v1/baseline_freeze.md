# PF 6 h -> 48 h pilot baseline freeze

This report is for the user-selected exact commit 6b69895af2d1b86b99c57c5479ff767349c61efe
(codex/pf-zero-mode-restart-provenance-v1). The main worktree is deliberately
detached at that commit because the branch name is checked out by the separate
runtime worktree; no unrelated dirty files were changed.

The accepted scope is PF-only composition plus PF_CONSERVED_Y_ZERO_MODE_V1.
GP, RSMD, scheduled sources, GP birth/release/growth, direct beta injection, and
elasticity are OFF. The only source change made for this goal is the validation
gate allowing an explicitly materialized --init-mode raw_fields state at t=0;
VTK continuation remains forbidden and all later continuation uses the
checksummed zero-mode checkpoint.

The new cluster source root is:
/data/home/luozhiheng/tmp/codex_pf_zero_mode_6h48h_20260729_v1

The pilot output root is:
/data/home/luozhiheng/tmp/pf_6h48h_pilot_zero_mode_20260729

Cluster jobs: target 72670, dependency tail 72671. The target is the only
active job for this goal.

The unrelated untracked report directories already present in the worktree
were preserved. No commit or push was created.

## Accelerated continuation addendum

The initial diagnostic-every-step chain (job 72670) was stopped after the
12 h checkpoint. Job 72676 resumed from that checkpoint in the unique output
root `/data/home/luozhiheng/tmp/pf_6h48h_pilot_zero_mode_fast_20260729` and
completed 12 h→48 h. The start checkpoint SHA-256 was identical in the two
roots. The diagnostic AB qualification (job 72674) established bytewise-equal
continuous/restart states for diagnostic intervals 1 and 256, so the
accelerated continuation changes cadence only, not the accepted PF state.
