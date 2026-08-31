# KWN–PF thermodynamic contract and four-bucket state closure v1：最终验收

## 顶层状态

`FAIL_PF_SMOKE`。

唯一阻断原因是本工作站没有可用于此合同的 CUDA toolchain 或受控 PF binary；A–E 没有启动，不能由 host controls 代替。

## 对十二个验收问题的回答

1. **Historical as-run authority 是否恢复？** 否，`HISTORICAL_AS_RUN_AUTHORITY_UNRECOVERED`。本机可读到 downstream full-radius lists（001 REF/BROAD 121 条、004 NARROW 165 条），但它们的 raw checkpoint paths 不在本机，且 P1/14-case 均标为 conditional/not-formally-registered 或 `RETRYABLE_CASE_FAILURE`；没有唯一完整 as-run contract。
2. **Validation contract hash 是什么？** `d0ff02973ab0f737043e1a40d4f69893a469cbfe2bc4cd22f9e6a410bd0b1333`。
3. **PF 和 KWN 的 G、mu、solvus、D、curvature correction 是否一致？** Host probe 与 Python contract 一致：100 行最大相对误差 `3.657021380996818e-12`，solvus 最大绝对误差 `0`，两侧 beta driving-force direction 正确；状态 `PASS_THERMO_VALIDATION_HASH_BIND`。这不包含 CUDA elasticity/interface runtime replay。
4. **PF 是否能持久保存 GP 与 sub-grid beta？** V6 host checkpoint 能持久保存冻结的 compact auxiliary PSD、两库存、validation-contract 和 package identity；状态 `PASS_PF_ZERO_MODE_CHECKPOINT_PROVENANCE_V2_TO_V6_AUX`。它尚非已执行的 CUDA checkpoint/restart。
5. **zero-aux 修改是否保持旧 PF 行为？** Host field control 中 source `phi`/`xB_alpha` 未变，重映射 `xB` 最大差 `4.336808689942018e-18`；它不是运行轨迹的 bitwise PF 对比。
6. **four-bucket checkpoint/restart 是否通过？** Host serialization/readback 通过，V2–V5 backward reads 保留；V5 active auxiliary restart 因缺 package identity 被拒绝。CUDA restart 未运行。
7. **beta-only KWN 与 PF 方向是否一致？** 尚未证明。仅 KWN `t=0` occupied classes 对同合同 curvature sign 自洽（7.963658 nm dissolution；9.605316/10.408712 nm growth）；`PASS_BETA_ONLY_CODE_DIRECTION` 只是该子检查。KWN 在 `0.39317699499770825 h` strict positivity guard 停止，PF 未运行，整体为 `PARTIAL_BETA_ONLY_DIRECTION_T0_ONLY_KWN_RUNTIME_INCOMPLETE`。
8. **mean-field 与 spatial/elastic PF 的差异可量化多少？** `NOT_MEASURED_NO_CUDA`；没有 PF trajectory，不能量化。
9. **actual fixture-conditioned KWN package 是否可行？** Host package 可行：`PASS_FIXTURE_CONDITIONED_HANDOFF_V2`，nonzero GP、fixed resolved beta、matrix 与 subgrid ledger 闭合，raw init 可在新目录 materialize；case E CUDA feasibility 尚未证明。
10. **v1 package 为什么不能直接使用？** 它的 resolved-beta bucket 为零，而六颗粒 fixture 已有非零 resolved β；直接相加会 double count，且 v1 不含此轮 V6 auxiliary/package-bound closure。
11. **96³ smoke 是否出现 matrix pulse、seed loss 或质量漂移？** `NOT_MEASURED_NO_CUDA`；所有 A–E × 0/6/48 h 均为 `NOT_RUN_NO_CUDA_OR_PF_BINARY`，不能称为无异常。
12. **下一步是否有资格进入 GP release、effective-CNT refinement 或更大 PF case？** 都不具备资格。先在同一合同、fixture、source/package provenance 下运行 A–E CUDA smoke，再在不调物理参数、不 clamp 的前提下定位 KWN positivity failure。

## 紧凑 gate 摘要

| Gate | 状态 |
|---|---|
| historical as-run | `HISTORICAL_AS_RUN_AUTHORITY_UNRECOVERED` |
| thermo | `PASS_THERMO_VALIDATION_HASH_BIND` |
| four bucket | `PASS_PF_FOUR_BUCKET_STORAGE`（仅 host） |
| historical PSD | `HISTORICAL_12H_PSD_NOT_RECOVERED` |
| beta-only | `PARTIAL_BETA_ONLY_DIRECTION_T0_ONLY_KWN_RUNTIME_INCOMPLETE` |
| fixture-conditioned handoff | `PASS_FIXTURE_CONDITIONED_HANDOFF_V2`（仅 host/package） |
| A–E smoke | `NOT_RUN_NO_CUDA_OR_PF_BINARY` |

四库存 fixture-conditioned residual 为 `1.4878348096565177e-16`；matrix→GP→matrix transfer residual 为 `3.862297964393128e-14`，forward/reverse matrix inverse residual 分别为 `3.712771878204502e-16` 与 `3.675644159422457e-16`；double-count check 为 `PASS_NO_DOUBLE_COUNT`。

## 证据入口

- [validation contract](../../contracts/pf_kwn_validation_contract_v1.json)
- [thermo parity summary](../../outputs/kwn_pf_state_closure_v1/thermo_cross_language_summary.json)
- [four-bucket summary](../../outputs/kwn_pf_state_closure_v1/four_bucket_storage_control_summary.json)
- [fixture-conditioned validation](../../outputs/kwn_pf_state_closure_v1/kwn_pf_handoff_fixture_conditioned_v2/validation_report.json)
- [PF smoke status](../../outputs/kwn_pf_state_closure_v1/pf_smoke_trajectories.csv)

当前只证明了 hash-bound thermodynamic parity、host-side four-bucket storage/compact handoff/可物化 raw init，以及 source-level auxiliary import bridge；当前仍未证明 actual CUDA 96³ PF A–E smoke、PF–KWN trajectory/direction consistency 或 historical-production consistency；不允许进入局部 GP release 开发。
