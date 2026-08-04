# Historical accepted-field mechanics replay provenance

Status: `PASS_HISTORICAL_MECHANICS_ACCEPTED_FIELD_REPLAY_AUDIT_V1`.

A, B, and C each use one frozen Method-1 production authority. The 6 h field is the immutable initial fixture; 12-48 h fields are read from the exact audited checkpoints. Every checkpoint is verified against its authority audit before replay.

The replay advances neither PF fields nor simulation time and writes no checkpoint. Historical authorities are read-only. The only qualified historical initialization is `checkpoint_warm`; the 6 h fixture uses the synchronized pre-update mechanics diagnostic.

Parameter SHA-256: `ecbdd0ac070bdf5e5d214322b5248a08f5ca5dd4e0670427513f5ef977ea977a`.

## Source hashes

- `cuda_kernels.cu`: `0d36841845992e22ff4db3b4903200f49dc2b3ad1d42ab415b27477d72217f5a`
- `main_cuda`: `68f94f3128c73c5f9316543b4234b35ab00b13638983bdf7223b976b33f3be36`
- `main_cuda.cu`: `d4809ca43e04034eeaa3df92df3dc93a0a196530ac34014aad850bedf3d20453`
- `pf_zero_mode_checkpoint.cpp`: `e9c65556ab68ba231493924b4c90b36cc0c3a69430930f2b540774f6c00c6003`
- `run_historical_mechanics_accepted_field_replay_v1.sh`: `085555f87e0701a3c026e5ec16f198dad21722ee2c4fcfa3049d4e88e6e077a9`
