# PF 四库存状态设计

## 状态合同

完整 PF box 的 pseudo-binary B 绝对库存定义为：

`Q_B_total = Q_B_matrix + Q_B_beta_resolved + Q_B_GP + Q_B_beta_subgrid`。

四项均以完整 box 的 `mol B` 保存；输出时才用 box volume 换算 `mol B m⁻³`。`Q_B_matrix` 和 `Q_B_beta_resolved` 必须由现有 fixture fields 精确积分：

- `Q_B_matrix = sum((1-h(phi))*xB_alpha)*voxel_volume/Vm_alpha`；
- `Q_B_beta_resolved = sum(h(phi)*v_B)*voxel_volume/Vm_beta`。

GP 与 sub-grid beta 是 compact PSD 中的绝对库存，不能把 `mol B m⁻³` 直接写进 PF `xB_alpha`。

## 持久化与冻结语义

`pf_zero_mode::AuxPopulationState`（state schema V1）表达 GP/sub-grid beta 的冻结状态、合同/来源 handoff hash、package handoff hash、单位、两项库存、完整 PSD bins 及 provenance。当前 V6 checkpoint 同时保存 validation-contract 与 package identity；aux 存在时二者都必须一致。V2–V4 保持 legacy-zero-aux backward read；V5 可读取但 active auxiliary state 缺少 package identity，因而不得作为当前 active-aux restart 继续执行。

fixture-conditioned v2 package 另写 deterministic compact sidecar `auxiliary_population_state_v1.txt`（schema `PF_AUXILIARY_HANDOFF_V2_SIDECAR_V1`）。它携带 contract/source-handoff/package/fixture identity、四库存 ledger、冻结人口的 PSD bins 与以下固定声明：

`FIXTURE_CONDITIONED_PRESCRIBED_SOURCE_IS_NOT_A_GP_NUCLEATION_PREDICTION`

`main_cuda.cu` 的 source 为 fresh raw-field start 提供 sidecar reader，并要求 contract/source-handoff/package/fixture 四项 identity 与 raw-init metadata 相符；fresh auxiliary start 的超界 raw value 会 hard-fail，不会被 clamp。它继续禁止 auxiliary inventory 注入 `eta`、matrix 或 resolved field。该 bridge 仍未在本机 CUDA 编译或运行；这是 source-level binding/no-clamp 设计，而非 PF runtime evidence。

独立 C++ host reader 已通过 `PASS_PF_AUXILIARY_HANDOFF_V2_HOST_MATERIALIZER`：它拒绝 identity/ledger 不一致，并将 compact PSD 的 `n_per_m4 * ΔR` 显式换算为 checkpoint state 的 `number_density_m3`，保持 per-bin inventory、将连续谱 bin count 留为零。该结果只验证 sidecar→host checkpoint-state 的物化，不改变上述 CUDA 证据边界。

## 明确排除的动力学

auxiliary state 为 storage-only，不能写入 `eta`、`phi`、`xB_alpha`、chemical potential 或 beta seed；也没有每步 KWN、GP release、GP→beta conversion、new beta birth 或 dislocation path。既有 resolved β geometry 仅属于 `Q_B_beta_resolved`，不会从 compact KWN beta PSD 再生成。旧 legacy restart 也不能自动重签为当前 V6 package-bound validation checkpoint。

该设计验证的是 host persistence/storage contract；actual CUDA field evolution、0/6/48 h frozen-aux invariance 与 PF restart trajectory 需要单独 smoke。
