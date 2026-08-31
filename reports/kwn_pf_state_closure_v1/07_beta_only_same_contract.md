# Beta-only same-contract control

## 已得到的 code-level 证据

`PASS_BETA_ONLY_CODE_DIRECTION` 只适用于 KWN 的 `t=0` occupied size classes。它检查的是：从 SHA-bound 六颗粒 fixture 提取的 resolved β inventory/PSD 在 beta-only KWN 中的初始速率符号，是否与同一 validation contract 的 Gibbs–Thomson curvature sign 一致。

| KWN grid class | 合同 curvature sign | KWN `t=0` rate sign |
|---:|---|---|
| 7.963658 nm | dissolution | dissolution |
| 9.605316 nm | growth | growth |
| 10.408712 nm | growth | growth |

该 KWN control 使用合同原始 `D=8.073255954803963e-18 m² s⁻¹`、`gamma=0.168 J m⁻²`、`Vm=4.1009e-05 m³ mol⁻¹`，GP/beta nucleation 均关闭，未使用 fitted `D_scale`。初始 fixture/KWN inventory relative residual 为 `1.554061176041917e-16`。为避开未占用的 legacy 0.25 nm bin 的无意义 exact Gibbs–Thomson 解，数值网格下沿提高到合同推导的 `4.724027182871601e-10 m`；最小初始 fixture 半径仍为 8 nm，未改变物理参数或粒径。

这不是 PF-vs-KWN direction comparison：没有实际 PF 的初始 force/rate 或 trajectory 可供比较。因此不能从该 gate 推断 PF 与 KWN 已在真实时间演化中同向。

## 未完成的 runtime 与比较

总体状态为 `PARTIAL_BETA_ONLY_DIRECTION_T0_ONLY_KWN_RUNTIME_INCOMPLETE`。在 step 87、`0.39317699499770825 h`，strict finite-volume update 触发 `SolverStateError: beta finite-volume update violated positivity or finiteness`。没有通过降低 CFL、调整 `D`、改物理参数或 clamp inventory 来掩盖失败；6 h checkpoint 与 48 h KWN trajectory 均未生成。

完整 CUDA PF 在本机未运行，所有 PF trajectory 为 `NOT_RUN_NO_CUDA_OR_PF_BINARY`。因此尚无 code-timescale gate，亦无 PF-vs-KWN 的 `N(t)`、`R_mean³(t)`、`1/N(t)`、`S_v(t)`、`f_beta(t)`、`xB_matrix(t)` 或 PSD Wasserstein comparison，不能量化 mean-field 与 spatial/elastic PF 的差异。

历史比较同样未运行：`HISTORICAL_BETA_ONLY_COMPARISON_NOT_RUN`，原因是 historical as-run authority 与完整 12 h PSD 均未恢复。
