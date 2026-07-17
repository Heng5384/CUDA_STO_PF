# Production candidate decision

## Decision

No new production candidate is selected.

1. V2 has no formally qualified relaxed transport gate
   (`FAIL_V2_MATERIAL_TRANSPORT_DEFECT`). G10 is diagnostic only.
2. V5 improves a 2000-step dt/4 diagnostic but still requires fallback; in the
   8000-step dt/16 holdout it records 84
   internal rejects and 82 fallback macros.
3. V5 dt/8 stops at step 3465;
   V5 dt/2 stops at step 831.
4. The 400^3 source allocation is over the workstation admission limit before
   FFT/context overhead.

`LEGACY_CURRENT` remains the default and the accepted dt/16 baseline. The new
mode remains default-off research code. The correct next action is to resolve
the formal residual-gate defect and the active-bound globalization root cause
before memory refactoring or 3D campaign work.
