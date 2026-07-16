# Active-manifold BDF2 baseline freeze

This goal starts from the already accepted event-safe T400 source-free baseline.
No physics parameter, tolerance, authoritative state, or checkpoint was changed
by the host candidate analysis.

| Contract | Frozen value |
|---|---|
| Physics | `pbte_ag2te_gp_coarse4_stoich_rd_v2` |
| Base integrator | `ctot_jichen_imex_bdf2_v1` |
| Event contracts | `BDF2_EVENT_PREFLIGHT_V1`, `BDF2_EVENT_BE_SUBCYCLING_V1` |
| Endpoint work | `BDF2_STABLE_ADMISSIBLE_ENDPOINT_WORK_V2` |
| Grid | `512x1x1`, `dx=1 nm`, `lambda=4 nm` |
| T | `400 C` |
| GP/source/elasticity | `OFF/OFF/OFF` |
| Workstation | `RTX 5080`, CUDA 12.9 |

## Current source hashes

| Asset | SHA-256 |
|---|---|
| `main_cuda.cu` | `815cafbba0912c55d3b8910ba66cca7a9329f656c01e279763b2c86a98e89b99` |
| `cuda_kernels.cu` | `a4d269067ab2de6059af1d48c333eda80aec43268c3f61036eab1a66e760b81a` |
| `cuda_kernels.h` | `3eb5cfa9ed35ddb533f43d5847ca270a0ac3a4ddb389770140e7fa7e7c425b53` |
| `pf_params.h` | `4ebeac11681af092de58f1e19a2bac6c615368659263a58e138e4b1678111cd1` |
| `bdf2_event_utils.h` | `9b7712f96d62fc1e3f388c44cc9bd8e38bd3ded76868ea7e15c49615b9429115` |
| `phase_kkt_utils.h` | `76603069f6c7045e5634212335b8dd24658ec8cfa430299f2591139d20460d19` |
| `active_manifold_bdf2_utils.h` | `e4fcce0513ee44fd462ebf711e247ffbc297cd73cefe3a26cf2f3039439489eb` |

## Frozen event state hashes

| Asset | SHA-256 |
|---|---|
| `reports/bdf2_event_v1/frozen_checkpoints/dt16/ctot_checkpoint_step005482_Ctot.raw` | `28772b82858676d5a530480d9e8c77b850e2bbd379431e67c28018501b0aaa4f` |
| `reports/bdf2_event_v1/frozen_checkpoints/dt16/ctot_checkpoint_step005482_Ctot_nm1.raw` | `6c2981018ded755f1894d1afb5d8beb7b5b50ea149d37b6b86353897caf2adf9` |
| `reports/bdf2_event_v1/frozen_checkpoints/dt16/ctot_checkpoint_step005482_phi.raw` | `085c591dd636207ab411fe5be31b8a6de64593eb7f601b2880dd3c4c1e9e4db1` |
| `reports/bdf2_event_v1/frozen_checkpoints/dt16/ctot_checkpoint_step005482_phi_nm1.raw` | `668dcb2d8317e53930ee3680d34767c9a003e62403008bb3256c9d9362d17072` |
| `reports/bdf2_event_v1/frozen_checkpoints/dt16/ctot_checkpoint_step005482_xB_alpha.raw` | `64c34df3d05b98728b2eec4104d3085e0267f3e24565e47269a8dd15eb53b98e` |
| `reports/bdf2_event_v1/frozen_checkpoints/dt8/ctot_checkpoint_step002957_Ctot.raw` | `fe11fd1be7bd5a30b46ec0afe382a1f07d7970ec24b6419abd293f3c247cf268` |
| `reports/bdf2_event_v1/frozen_checkpoints/dt8/ctot_checkpoint_step002957_Ctot_nm1.raw` | `c10adb6c09cccc1dd1e7fe4bee2196b3f53d05701b0ee2bc62f7cf04cdd04f65` |
| `reports/bdf2_event_v1/frozen_checkpoints/dt8/ctot_checkpoint_step002957_phi.raw` | `c1a0477b4384fd829a904ec4f88e943e2737f1308fefe01027394101c35ad0b8` |
| `reports/bdf2_event_v1/frozen_checkpoints/dt8/ctot_checkpoint_step002957_phi_nm1.raw` | `1dcd47790398a9aaec45a186bf9c6c1e745b3c4059c8130bcdecd826e113a7c9` |
| `reports/bdf2_event_v1/frozen_checkpoints/dt8/ctot_checkpoint_step002957_xB_alpha.raw` | `5708d59b5ebd36795707caa5ae41163a8dfd24655619bb0dcd03d39bd136d850` |

The prior 502-macro metrics and fine-reference hashes remain authoritative in
`reports/bdf2_event_v1/event_crossing_metrics.csv` and its frozen run manifests.

`baseline_preserved=true`
