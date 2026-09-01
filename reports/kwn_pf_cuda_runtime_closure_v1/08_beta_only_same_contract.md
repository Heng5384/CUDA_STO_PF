# 08 Beta-only KWN–PF same-contract trajectory

Status: `NOT_RUN_PREREQUISITE_FAIL_KWN_CONSERVATIVE_POSITIVITY`.

Although CUDA finished `PASS_CUDA_AE_SMOKE`, the KWN qualification is `FAIL_KWN_CONSERVATIVE_POSITIVITY` because P5 radius-grid convergence is false. The task contract permits the beta-only KWN–PF trajectory comparison only after both `PASS_CUDA_AE_SMOKE` and `PASS_KWN_CONSERVATIVE_POSITIVITY`.

No scalar-moment substitute, no D-scale fitting, and no incomplete KWN trajectory is used. Consequently growth/dissolution direction, timescale, PSD Wasserstein distance and mean-field gap are not claimed.
