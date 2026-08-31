# 模型边界与未证明事项

- validation contract 不是 historical as-run contract；不得用它重新解释既有 246³/400³ production。
- 合同内的 `C_alpha_voigt_GPa`、`C_beta_voigt_GPa` 和 eigenstrain 是 validation fixture 的记录，不是已验证的 CUDA elastic replay。KWN beta-only control 仍是 elastic-penalty 为零的 spherical mean-field approximation。
- `AuxPopulationState` 与 v2 compact sidecar 只保存冻结 GP/sub-grid beta PSD/库存；它们不提供 GP thermodynamics、GP release、GP→beta conversion、online coupling、beta birth 或 dislocation physics。
- V6 host checkpoint serialization、field inverse mapping、raw-init materialization 和 compact sidecar readback 不等于 CUDA runtime verification。source-level raw-init identity/no-clamp checks 亦尚未编译执行。缺少 `nvcc`/受控 PF binary 时，96³ A–E dynamics/restart smoke 一律为 `NOT_RUN_NO_CUDA_OR_PF_BINARY`。
- beta-only 只证明 KWN `t=0` rate sign 与同合同 curvature sign 一致；它没有实际 PF trajectory，且 KWN strict positivity guard 在 `0.39317699499770825 h` 停止。因此 mean-field 与 spatial/diffuse-interface/elastic PF 的差异尚未量化。
- historical 12 h full PSD 与唯一 as-run contract 未恢复；不得从 moments 伪重建，也不得把 96³ validation control 宣称为历史生产验证。
- fixture-conditioned GP 是 prescribed source，必须保留 `FIXTURE_CONDITIONED_PRESCRIBED_SOURCE_IS_NOT_A_GP_NUCLEATION_PREDICTION`；它不是物理 GP 成核模型。

当前顶层 smoke 状态不是 `PASS_KWN_PF_ONE_WAY_STORAGE_COUPLING_V1`，因此没有资格进入 local GP release、effective-CNT refinement 或更大 representative PF case。下一步应先在受控 CUDA 环境运行同一合同和 fixture 的 A–E smoke，并在不改物理参数的前提下定位 beta-only KWN positivity failure；GP release 仍在当前主研究范围之外。
