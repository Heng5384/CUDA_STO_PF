# Accepted-field mechanics replay qualification

Status: `PASS_MECHANICS_ONLY_ACCEPTED_FIELD_REPLAY_V1`

The online reference and both offline replays solve the same accepted
phi/xB field before any PF, source-field, time, or checkpoint advance.
The authority path is checkpoint-warm replay, matching historical V4
production checkpoints. Zero initialization is an additional sensitivity
test only and is never permitted for authority replay unless it independently
passes every field gate.

| Comparison | Energy rel. error | Max strain L2 | Max stress L2 | Source exact | Status |
|---|---:|---:|---:|---|---|
| online_vs_checkpoint_warm_replay | 0.000e+00 | 0.000e+00 | 0.000e+00 | True | PASS |
| online_vs_zero_initialized_replay | 1.494e-09 | 4.897e-05 | 1.416e-05 | True | FAIL |

The zero-initialized sensitivity did not pass the frozen field
tolerances and is explicitly rejected. No tolerance was relaxed;
all historical replays are therefore locked to checkpoint_warm.

Acceptance gates: energy relative error <= 1e-6; normalized L2
error <= 1e-5 for every strain and stress component; source fields
must be byte-identical and all no-advance flags must be true.
