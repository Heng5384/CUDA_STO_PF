# 6–8 h 高时间分辨形貌、15–16 h 合并追踪与运行时弹性能增量审计

## 结论

`PASS_INCREMENTAL_MERGE_MORPHOLOGY_AND_RUNTIME_ELASTIC_SCALAR_V1`

本次按增量方式补齐三个优先项，没有重跑整套 6–48 h：

1. 15–16 h 的异常身份变化已确认为追踪阈值上的 2→1 连通，并升级为持续的 merge-lineage group；
2. 使用同一 6 h 初态、同一物理参数和 `dt_code=0.02`，完成 6–8 h、每 256 步一帧的高时间分辨形貌审计；
3. PF-only 弹性路径已在运行时输出非零的平均/峰值弹性能标量和静水应力范围。

8 h 的 `phi`、`xB`、`xBtot` 与原始已完成运行逐字节一致，说明新增诊断和高频输出没有改变相场轨迹。

## 运行身份

| 项目 | 值 |
|---|---|
| 分支 | `codex/pf-dynamic-microstructure-audit-v1` |
| 基线 HEAD | `5bb0db2bb3321b70cb5f254ffd9ab738a3b3611d` |
| Cluster job | `72956`, `COMPLETED`, exit `0:0` |
| 网格 | 246³，`dx=1 nm` |
| 温度 | 380 °C |
| 弹性 | 开启 |
| GP | 关闭 |
| `dt_code` | 0.02 |
| 单步物理时间 | 0.9909260953431842 s |
| 运行区间 | 6–8 h，7266 步 |
| 场输出间隔 | 256 步 = 253.677 s = 4.228 min 物理时间 |
| 高频快照数 | 30 |
| 墙钟 | 主运行约 54 min 37 s；作业总计 57 min 27 s |
| 实测速率 | 约 0.449 s/步（含高频 VTK、检查点和诊断） |

## 15–16 h merge-aware 追踪

状态：`PASS_MERGE_AWARE_GROUP_TRACKING`

原先在 16 h 被 fail-closed 标记的身份异常不是单颗粒溶解或数值丢失，而是颗粒 46 与 68 在追踪阈值 `h>1e-4` 下成为同一连通体。现建立持续组：

`MG_15h_46_68_to_46`

| 时间 | 组状态 | 连通体数 | 组 h-体积 (nm³) | 组等效半径 (nm) |
|---:|---|---:|---:|---:|
| 14 h | 合并前双成员 | 2 | 33,987.4853 | 20.0945 |
| 15 h | 合并前双成员 | 2 | 36,205.8129 | 20.5225 |
| 16 h | 合并后的 child 46 | 1 | 38,097.2375 | 20.8738 |
| 17 h | 合并后连续追踪 | 1 | 40,541.9975 | 21.3111 |

重叠证据：

- parent 46 → child 46：43,318 voxels，覆盖 parent 的 100%，占 child 的 60.4426%；
- parent 68 → child 46：24,297 voxels，覆盖 parent 的 95.4433%，占 child 的 33.9022%；
- split edge：0；
- 15→16 h 组 h-体积变化：+5.2241%。

因此，粒子 68 不再计为“溶解消失”；其 lineage 在 16 h 后由合并组继续承载。后续多阈值审计进一步表明，16 h 只连通到 `h=0.001`，而 `h>=0.005` 的较强物质颈部到 17 h 才建立，所以 16 h 应解释为 diffuse-neck onset，不是已经完成实体并合。

## 6–8 h 高时间分辨形貌

状态：`PASS_HIGH_TIME_MORPHOLOGY_OBSERVATION_V1`

| 指标 | 6 h | 8 h | 相对变化 |
|---|---:|---:|---:|
| 颗粒数 | 96 | 59 | −38.542% |
| β h-体积分数 | 0.0239295450 | 0.0227653592 | −4.865% |
| 平均等效半径 (nm) | 9.54047 | 10.67676 | +11.910% |
| 半径 P10 (nm) | 8.49959 | 8.10386 | −4.655% |
| 半径 P90 (nm) | 10.63955 | 12.98596 | +22.053% |
| \(S_v\) (nm⁻¹) | 0.007425295 | 0.005941920 | −19.977% |
| \(M_6\) (nm³) | 5.36261 | 9.22645 | +72.052% |
| 阈值界面面积密度 (nm⁻¹) | 0.010931732 | 0.009041887 | −17.288% |
| matrix \(x_B\), `h<0.005` | 0.00622000 | 0.00740441 | +19.042% |

时间分辨结果显示：

- 6.000–6.634 h：颗粒数仍为 96，但 β 体积分数和平均半径先经历初态调整；
- 首次可分辨的颗粒数下降位于 6.634–6.705 h；
- 6.846–7.057 h 是较集中的损失区间，颗粒数由 93 降至 82；
- 7.057–7.973 h 继续阶梯式下降至 59；
- 7.973–8.000 h 颗粒数保持 59；
- 所有 30 帧均未触发意外 merge/split；
- 最大 h-体积闭合误差为 `5.14446e-7`。

基体 `xB` 并非单调变化：它在约 6.846 h 达到 `0.00777205`，随后回落到 8 h 的 `0.00740441`。这说明早期小颗粒消失释放溶质与存活颗粒继续吸收之间存在动态竞争，不能只用 6 h 与 8 h 两个端点解释。

## 运行时弹性能标量

状态：`PASS_RUNTIME_ELASTIC_SCALAR_EXTRACTED`

PF-only 运行现在在既有质量诊断节拍上输出：

- `mean_elastic_energy`；
- `max_elastic_energy`；
- `stress_hydro_min`；
- `stress_hydro_max`。

能量换算使用已冻结的

\[
\frac{12\gamma}{\lambda_{\rm sm}}
=\frac{12\times0.168}{4\times10^{-9}}
=5.04\times10^8\ {\rm J\,m^{-3}}.
\]

运行时共有 28 个弹性能采样点，覆盖 6.070–8.000 h：

| 指标 | 首个采样（6.070 h） | 末采样（8.000 h） | 变化 |
|---|---:|---:|---:|
| 平均弹性能密度 (J/m³) | 567,971.5 | 539,765.4 | −4.966% |
| 峰值弹性能密度 (J/m³) | 21,082,694.6 | 18,640,218.3 | −11.585% |
| 盒内总弹性能 (J) | \(8.45536\times10^{-15}\) | \(8.03545\times10^{-15}\) | −4.966% |

平均弹性能在约 6.846 h 达到本窗口最低值 `533,852.2 J/m³`，随后略有回升；这与 β 体积分数最低点及基体 `xB` 峰值处于同一时间邻域。该对应关系是动力学相关性，不能单独解释为因果机制。

`stress_hydro_*` 保留求解器的应力标度；本报告不在缺少独立应力单位合同的情况下擅自换算成 Pa。

## 守恒、零模与轨迹不变性

- `pf_zero_mode_status=PASS_PF_CONSERVED_Y_ZERO_MODE_V1`
- 最终目标质量：`4.46608079999958049e+05`
- 最终质量：`4.46608079999962298e+05`
- 最终平均质量误差：`2.85428730712829079e-16`
- 诊断点最大单步总质量变化绝对值：`1.96718e-15`
- 最终 zero-mode lambda：`-2.83735e-7`
- 第一步 raw-fixture 接管的最大 \(|lambda|\)：`0.0469574`，之后快速下降；
- zero-mode 全程不需要 bisection；
- 8 h `phi` SHA-256：`6a8bfa68ec111b82a43072e8e0edb275fba51392a438ebaff5898c083244a69f`
- 8 h `xB` SHA-256：`9b538168ef07d4da8baff69b06105871464f4dc2d759f6743e3fafb1349671be`
- 8 h `xBtot` SHA-256：`51a3421b882bc6296980f08252d6dca24db87b192be8347d43d63041950bdce9`
- 三个 8 h 场均与原运行逐字节一致。

## 边界与下一步

本次没有重跑 8–48 h，因此历史 48 h 轨迹本身仍没有 8 h 以后的运行时弹性能时间序列。已经闭合的是：

- PF-only 弹性能标量的代码路径；
- 6–8 h 的真实运行资格；
- 15–16 h 合并事件的 lineage 解释。

以后新生产运行会自动携带该弹性能标量。若论文确实需要 15–16 h 合并前后的弹性能瞬态，应从可验证检查点另做局部窗口；现有运行只保留最终覆盖式检查点，不能在不重新积分 12–15 h 的情况下无损补回该标量。

## 最终标记

```text
merge_aware_status=PASS_MERGE_AWARE_GROUP_TRACKING
merge_group_id=MG_15h_46_68_to_46
merge_parent_ids=46,68
merge_child_id=46
split_edge_count=0

high_time_morphology_status=PASS_HIGH_TIME_MORPHOLOGY_OBSERVATION_V1
high_time_snapshot_count=30
high_time_physical_spacing_min=4.22795
particle_count_6h=96
particle_count_8h=59

runtime_elastic_scalar_status=PASS_RUNTIME_ELASTIC_SCALAR_EXTRACTED
runtime_elastic_sample_count=28
eight_hour_field_identity=PASS_BYTEWISE
pf_zero_mode_status=PASS_PF_CONSERVED_Y_ZERO_MODE_V1

full_48h_rerun=false
cluster_job=72956
cluster_job_state=COMPLETED
cluster_job_exit=0:0
final_status=PASS_INCREMENTAL_MERGE_MORPHOLOGY_AND_RUNTIME_ELASTIC_SCALAR_V1
```
