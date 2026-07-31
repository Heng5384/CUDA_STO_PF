# Unit and qualification tests

The committed host checkpoint test passed on cluster:
PASS_PF_ZERO_MODE_CHECKPOINT_PROVENANCE_V1

It covers round-trip identity, backend/time-level/reaction selector mismatch,
and payload corruption rejection. The existing cluster qualification at the
same exact commit passed continuous/restart byte equality on 64^3 and the
large-grid performance matrix on 400^3 and 512^3.

The new raw-fixture preflight passed on cluster job 72668:
PASS_PF_6H48H_RAW_FIXTURE_ZERO_MODE_PREFLIGHT_V1

It covers initial raw-field loading, zero-mode mass closure, 64-step
checkpoint/restart equality to step 128, and a dt/2 audit.
