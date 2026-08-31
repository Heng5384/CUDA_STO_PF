# Fixture-conditioned KWN handoff v2

## 结论与范围

fixture-conditioned v2 是六颗粒 96³ validation fixture 的 host-side、四库存 package control。其 validator/readback 状态为 `PASS_FIXTURE_CONDITIONED_HANDOFF_V2`，并以 `PASS_NO_DOUBLE_COUNT` 约束既有 resolved β geometry 只计一次。它不等同于 case E 已在 CUDA PF 上成功启动或运行。

所有非零 GP 的语义固定为：

`FIXTURE_CONDITIONED_PRESCRIBED_SOURCE_IS_NOT_A_GP_NUCLEATION_PREDICTION`

它是 prescribed-source storage allocation，不是 GP 成核预测，也没有采用 set 47 effective-CNT point。

## package 内容与守恒映射

v2 从可读的六颗粒 96³ host profile fields 构建；profile manifest/raw field 在读取前做 SHA-256 identity 验证，dense raw fields 不复制进 Git。package 位于：

`outputs/kwn_pf_state_closure_v1/kwn_pf_handoff_fixture_conditioned_v2/`

当前重建 package 的身份为：

| 字段 | 值 |
|---|---|
| contract hash | `d0ff02973ab0f737043e1a40d4f69893a469cbfe2bc4cd22f9e6a410bd0b1333` |
| fixture hash | `f1247cb66419af764b97de2f7843fc6de2049459d78550bc603edd8e88d9134f` |
| package hash | `5ffc4ab70afde89e7b23bfe642cdfba40034f60f20f4a42f7e3fcbcace99f3a8` |

`metadata.json` 的 `source_handoff_hash`、`auxiliary_sidecar_sha256`、array/hash manifest 与 `validation_report.json` 一起构成其余可复算 provenance；不得再引用 contract/sidecar 更新前的旧 package hash。

它包含 `metadata.json`、compact `arrays.npz`、`ledger.csv`、`validation_report.json` 与 `auxiliary_population_state_v1.txt`。sidecar 保留 GP/sub-grid PSD bin、四库存 ledger 和完整 identity；其语义为 `COMPACT_FROZEN_STORAGE_REQUIRES_SEPARATE_RAW_FIELDS`。

Python package roundtrip 与独立 C++ host reader 均已验证 sidecar 的 deterministic identity/ledger/PSD materialization；后者状态为 `PASS_PF_AUXILIARY_HANDOFF_V2_HOST_MATERIALIZER`。这只说明 compact package 能安全交给 host checkpoint-owned auxiliary state，不是 GPU runtime claim。

矩阵采用 `baseline + delta_C_relaxation/(1-h(phi))` 的 inverse-storage mapping：保留 `phi`、既有 resolved β geometry 与 `delta_C_relaxation`，只调基准值；不对全场覆盖常数、不会将 GP 加入 matrix、且不使用 clipping。resolved β 从 source field 精确积分，并在分配剩余 matrix/GP/sub-grid 库存前固定，因此不由 compact KWN PSD 再生成。

## raw-init 与 CUDA 边界

validator 的 `pf_raw_initialization_allowed=true` 仅说明当前四库存/field mapping 的 host-side feasibility 条件成立；v2 package 本身的 `pf_raw_initialization_emitted=false`，不携带 dense PF raw fields。单独的 `scripts/materialize_fixture_conditioned_kwn_pf_handoff_v2_raw_init.py` 可在显式指定的新目录中，从 real hash-validated profiles 与 feasible compact package materialize `phi_init.raw.f64`、`xB_init.raw.f64`、raw-init meta/provenance；它拒绝 synthetic/覆盖路径，并明确标记 `NOT_RUN_RAW_INITIALIZATION_MATERIALIZATION_ONLY`。compact sidecar 仍不是 GP/β-subgrid field materialization。

`main_cuda.cu` 的 source 已提供 fresh raw-field start 的 `--pf-auxiliary-sidecar` 读取入口。它要求 contract/source-handoff/package/fixture metadata identity 相等，拒绝 meta-path 不一致或 fresh auxiliary start 的 raw-field clamp，并检查 materialized mean `xBtot`；它不声称对 dense raw file 或 sidecar 做 runtime SHA rehash。该 source bridge 目前仅为 `SOURCE_INTEGRATED_UNCOMPILED_NOT_RUN`；在本机没有已运行的 CUDA PF binary，故没有实际 CUDA raw-init、case E 0/6/48 h trajectory、GPU checkpoint/restart、matrix pulse 或 seed-loss 结果。case E 的实际 smoke 状态保持 `NOT_RUN_NO_CUDA_OR_PF_BINARY`。
