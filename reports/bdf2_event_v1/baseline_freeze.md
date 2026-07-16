# BDF2 active-set event baseline freeze

## Scope

This freeze is workstation-only and source-free.  It preserves the accepted
T400 coarse4 IMEX-BDF2 model before any event-handling or endpoint-audit edit.

| Item | Frozen value |
|---|---|
| Git HEAD | `d8e836566829eb3458641346cdaca6a2ddf3ed60` |
| Branch | `codex/pf-ctot-production-candidate` |
| Model | `pbte_ag2te_gp_coarse4_stoich_rd_v2` |
| Numerics | `ctot_jichen_imex_bdf2_v1` |
| Transport | `mimetic_shared_face_v1` |
| Phase solver | `semismooth_pdas_v1` |
| Energy/work contract | `BDF2_ENDPOINT_DISCRETE_WORK_V1` |
| T | 400 C |
| Grid | 512 x 1 x 1, dx = 1 nm |
| Interface | lambda = 4 nm |
| Source/GP/elasticity | OFF/OFF/OFF |
| Retry/adaptive dt | OFF/OFF |
| CUDA | 12.9, compiler build 36037853_0 |
| Host compiler | GCC 13.3.0 |
| GPU | NVIDIA GeForce RTX 5080, driver 580.95.05 |

## Frozen hashes

| Asset | SHA-256 |
|---|---|
| `main_cuda` | `00855adfcb13864b8ac29a7d1cefc8fb90d4bc2c74604642be6aa210ac5ffc98` |
| `main_cuda.cu` | `08c87c390be1b696bde8cebb3f2bd0bc70b64841464038a382507f00a81ace7e` |
| `cuda_kernels.cu` | `a4d269067ab2de6059af1d48c333eda80aec43268c3f61036eab1a66e760b81a` |
| `cuda_kernels.h` | `3eb5cfa9ed35ddb533f43d5847ca270a0ac3a4ddb389770140e7fa7e7c425b53` |
| `pf_params.h` | `b42e6f9c83febf727b55dc12102572d08e1b970b36cbe7f0f0ebcbf19d01c778` |
| `thermo_utils.h` | `7d1fb79cd45b94f6ce50cd85f4e4b37f005838ff9c09fcd66818f8001bb84f99` |
| `phase_kkt_utils.h` | `76603069f6c7045e5634212335b8dd24658ec8cfa430299f2591139d20460d19` |
| dt/8 params | `5259ac5eb44876c381e84caf4bd68291ff415a94b33c9a56da45e6517530210f` |
| dt/16 params | `d70095f9b80d62c20c3ab7301c03129ba998c92461a3c899259d621219e2a095` |

The full source and solver-parameter manifest remains in
`reports/T400_longtime_v1/baseline_manifest.json`.

## Exact reproduction

The unmodified frozen binary was run twice per dt on `workstation-tail`.

| Case | Last accepted | Rejected step | Frozen physical time | Reproduced predicate |
|---|---:|---:|---:|---|
| dt/8, `dt=0.000390625` | 2957 | 2958 | 47.50788441422185 s | one extrapolated-context cell; same-dt BE residual `5.135121606042772e-12` |
| dt/16, `dt=0.0001953125` | 5482 | 5483 | 44.03757564402975 s | transport `9.921512107386807e-13` PASS; endpoint chain terms NaN |

The repeated step numbers, residuals, and rejection classes agree with the
original long-time evidence.  Baseline reproduction therefore passes.

## Frozen checkpoint locations

- dt/8 accepted step 2957:
  `reports/T400_longtime_v1/coarse4_runs/growth_N512_shift0_dt8_pre_event_freeze/run/`
- dt/16 accepted step 5482:
  `reports/T400_longtime_v1/coarse4_runs/growth_N512_shift0_dt16_pre_event_freeze/run/`
- dt/8 rejected-trial cell diagnostics:
  `reports/T400_longtime_v1/coarse4_runs/growth_N512_shift0_dt8_event_repro/run/`
- dt/16 rejected-trial cell diagnostics:
  `reports/T400_longtime_v1/coarse4_runs/growth_N512_shift0_dt16_event_repro/run/`

Each accepted checkpoint contains authoritative `Ctot`, `phi`, their `nm1`
history fields, reconstructed `xB_alpha`, and versioned metadata.  Rejected
trial diagnostics are captured before rollback; the accepted state was restored
without mass loss, clipping, projection, or accepted-state nonfinite values.

## Preservation decision

`baseline_preserved=true`

No physical parameter, hard tolerance, seed/source path, or accepted-state
history was changed during this stage.
