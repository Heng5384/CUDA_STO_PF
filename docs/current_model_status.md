# 当前模型与生产状态

> 同步时间：2026-08-04 03:34 CST
>
> 权威来源：[`PROJECT_CORE_MEMORY.md`](../PROJECT_CORE_MEMORY.md)
> 若本报告、历史报告或运行目录中的旧状态文件与核心记忆冲突，以核心记忆的最新章节为准。

## 1. 当前总状态

```text
WORKSTATION_246CUBE_MAIN_PRODUCTION=NOT_COMPLETE_NO_ACTIVE_PROCESS
CLUSTER_400CUBE_PSD_LADDER=RUNNING_ARRAY_PARTIAL
FINAL_6_TO_48H_PRODUCTION_AUTHORITY=NOT_AVAILABLE
```

这三个状态是当前项目级结论：已有多项数值、机制和接口资格通过，但新的
400³多PSD/空间/密度生产集合尚未完成，因此不能把局部PASS、静态fixture
资格或运行中的作业写成最终6–48 h生产权威。

## 2. 当前研究与物理边界

当前主线研究PbTe–Ag₂Te在给定6 h实验约束后的条件微观结构演化及其晶格
热输运响应。模型包含已有resolved β颗粒的溶解、长大、粗化和可能的物理
合并，不预测绝对首次β成核时间。

主线强制关闭：

- GP、GP Birth和GP release；
- 新β成核和外部物质源；
- 为匹配48 h结果进行的案例专用参数调整；
- 从弹性场、析出物密度或eigenstrain反演位错密度。

冻结的基础数值合同包括`dx=1 nm`、`lambda_sm=4 nm`、周期固定胞、
Ji–Chen守恒零模、checkpoint/restart provenance，以及单取向eigenstrain
合同。任何扩展物理必须与该基线分开定级。

## 3. 已验收的数值和初态能力

以下能力已有独立验收证据：

- 守恒PF、质量闭合、零模和checkpoint/restart；
- 弹性warm-start＋残差控制求解器；
- hash-pinned弹性target-profile库及其工程资格；
- 守恒、无重叠、多粒子条件接管初态；
- 主要物理观察量的生产dt收敛；
- 246³三个随机条件初态的短资格、身份和restart审计；
- merge-aware颗粒谱系、总体PSD和单粒子轨迹的分级审计。

代表性状态包括：

```text
PASS_PF_ELASTIC_TARGET_PROFILE_LIBRARY_ENGINEERING_V1
PASS_MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1
PASS_PRODUCTION_DT_OBSERVABLE_CONVERGENCE_V1
PASS_246CUBE_THREE_SEED_LIBRARY_HANDOFF_SHORT_QUALIFICATION_V1
PASS_ELASTIC_WARM_START_RESIDUAL_V1
```

条件接管初态不声明多颗粒共同化学—弹性平衡。初始弹性重构从6 h开始即
计入真实物理时间，不做预松弛后重新标时。

## 4. 246³条件路径及机制结论

历史单盒有弹性6–48 h轨迹和Method-1 A/B/C三条条件路径提供了有效的
机制与接口证据。merge-aware审计和条件生产overlay曾达到：

```text
PASS_MIXED_OSTWALD_AND_COALESCENCE_COARSENING_V1
PASS_246CUBE_6H48H_CONDITIONAL_PRODUCTION_V1
PASS_246CUBE_HOURLY_MERGE_DISSOLUTION_ABC_ENSEMBLE_V1
```

可支持的科学表述是：体系以弹性修正的Ostwald粗化为主，同时包含少量
diffuse-neck合并。不能宣称纯经典LSW，因为6–8 h有明显弹性初态重构，
后续仍有少量净β生长，并存在已确认的merge-lineage。

246³盒子在48 h的resolved颗粒数量很少，高阶矩容易由少数大颗粒主导。
这些路径可以证明条件演化机制，但不足以单独提供稳健的统计生产权威；
这也是多随机初态和更大400³盒子仍然必要的原因。

## 5. 晶格热输运状态

PF到输运的正式接口输出：

- 完整PSD和数密度`Nv`；
- 界面面积密度`Sv`；
- 六阶矩`M6`；
- matrix Ag浓度；
- 形貌、取向和时间信息。

已通过的工程与数值状态包括：

```text
PASS_PF_FULL_PSD_NO_DISLOCATION_TRANSPORT_INTERFACE_V1
PASS_PF_FULL_PSD_NO_DISLOCATION_TRANSPORT_ENSEMBLE_V1
PASS_PF_METHOD1_ABC_HOURLY_FULL_PSD_NO_DISLOCATION_DESCRIPTOR_SUFFICIENCY_V1
```

这些PASS不等于实验绝对热导率复现。当前无位错条件路径的外部实验比较为：

```text
yu_no_dis_pf_interface_status=PASS_NUMERICAL_AND_ENGINEERING
sheskin_external_comparison_status=FEASIBLE_AND_REQUIRED
current_absolute_experiment_status=FAIL_HIGH_BIAS
```

resolved颗粒密度对比和粗化本身不能解释实验6–48 h约21%的热导率恢复；
全局密度扫描已给出`NO_GO_RESOLVED_DENSITY_CONTRAST_GLOBAL_RANGE`。位错散射
只能作为`DISLOCATION_OFF`、`DISLOCATION_FIXED_EXTERNAL`或
`DISLOCATION_SENSITIVITY_BAND`外部条件处理。

Yu 2024的AQ公开模型复现通过，但公开48 h曲线与严格S13/表S2参数不闭合。
约0.119的等效位错散射强度仅是反演诊断，不能解释为作者拟合值，也不能
移植到本项目样品。

## 6. 当前材料级阻塞和NO-GO结论

- 若原样继承V2的AQ经验背景，V3绝对端点分支为
  `NO_GO_V3_IF_V2_BACKGROUND_INHERITED`。
- 宽诊断包络可满足端点必要条件，但正式有来源材料包络仍为
  `BLOCKED_V3_WIDE_SOURCED_MATERIAL_ENVELOPE_UNAVAILABLE_NUMERICAL_CONTROL_HAS_LEVERAGE`。
- 当前综合状态为
  `PASS_V3_ABSOLUTE_ENDPOINT_NECESSARY_CONDITION_MATERIAL_AUTHORITY_BLOCKED`；
  不能升级为材料可行性PASS。
- 在开展昂贵atomistic campaign前，需要独立资格化高温
  `C11/C12/C44`、branch damping和取向关系材料合同。

## 7. 400³生产集合的实时边界

截至同步时间，cluster数组`74232`处于部分运行状态：

```text
21/21 fixture static qualification PASS
9/21  short qualification PASS
0/21  final 6–48 h production PASS
```

运行目录为：

```text
/data/home/luozhiheng/tmp/pf_400cube_psd_spatial_density_ladder_production_v1_20260803
```

历史上，冻结profile库与一个精确目标h-volume/极严容差合同之间曾得到
`BLOCKED_PROFILE_LIBRARY_RANGE`。该数学边界不得被静默缩放、插值或修改
profile绕过。当前运行数组只有在其manifest明确记录后来获准且兼容的库存
合同，并逐case通过最终segment、restart、6–8 h和6–48 h分析门后，才能
形成production authority。

## 8. Workstation状态

同步时workstation没有活动的`main_cuda`或GPU compute进程。旧B/C目录中的
`RUNNING_246CUBE_6H48H_PRODUCTION_V1`状态文件最后更新于2026-07-31，实际
输出只到早期segment，不能解释为完成。已完成的短资格、restart、输运描述量
或mechanics replay任务也不能替代B/C完整6–48 h生产结果。

## 9. 历史组分架构状态

旧报告中的Mode L/X/Q是历史组分更新架构诊断，不再代表当前项目总状态：

| Mode | 历史定级 | 当前解释 |
|---|---|---|
| L: `lagged_rhs` | Failed diagnostic reference | 保留作失败机制和回归对照。 |
| X: `x_transport_projection_split` | Bounded diagnostic, not accepted | 全局投影破坏局部相存储语义。 |
| Q: `q_transport_projection_split` | Formula-level candidate, runtime not accepted | 保留作公式和回归目标，不是生产验收。 |

S3/RSMD源事务属于历史已验证组件，但当前主线继续冻结：

```text
S3_source_component_frozen=true
S3_reintegration_allowed=false
```

## 10. 下一阶段准入条件

1. 等待并逐case审计400³数组，不把`RUNNING`当成PASS；
2. 核对运行manifest与冻结profile/库存合同，关闭13.37与13.38之间的合同解释；
3. 形成多PSD、多空间seed和密度变化的统计量及有限尺寸敏感性；
4. 将通过生产权威门的完整PSD接入无位错/外部位错输运条件路径；
5. 在材料来源充分前，不升级V3材料结论，也不启动完整昂贵atomistic campaign。

## 11. 主要证据入口

- [`PROJECT_CORE_MEMORY.md`](../PROJECT_CORE_MEMORY.md)：研究边界、冻结合同和最新状态；
- [`reports/pf_246cube_method1_production_authority_v1/`](../reports/pf_246cube_method1_production_authority_v1/)：246³条件生产overlay；
- [`reports/pf_method1_abc_hourly_no_dislocation_transport_v1/`](../reports/pf_method1_abc_hourly_no_dislocation_transport_v1/)：A/B/C逐小时full-PSD无位错输运；
- [`reports/v3_absolute_endpoint_feasibility_audit_v1/`](../reports/v3_absolute_endpoint_feasibility_audit_v1/)：V3端点和材料权威边界；
- [`reports/pf_400cube_psd_spatial_density_ladder_production_v1/`](../reports/pf_400cube_psd_spatial_density_ladder_production_v1/)：400³生产集合及资格记录。
