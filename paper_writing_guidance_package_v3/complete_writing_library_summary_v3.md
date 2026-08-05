# PbTe–Ag₂Te V3 写作库完整总结

## 1. 当前冻结状态

- 写作库状态：`CONDITIONAL_PASS_PENDING_PRODUCTION_AUTHORITY`。
- 科学故事已经统一，可以用于 Introduction、Results、Discussion 和 Conclusions 的连续写作。
- Case 001–003 已有完整 48 h checkpoint 和可复现的条件性分析，但正式 merge-aware production-authority 后处理尚未冻结。
- 本总结是 V3 的完整入口；短版决策见 `latest_writing_library_executive_summary.md`，逐条证据见 `claim_evidence_matrix_v3.csv`。

这意味着：论文的中心思想已经成立，但当前数字和最强因果措辞仍需通过 P0 证据门槛后才能作为最终投稿结论。

## 2. 一句话故事与主 punchline

### 一句话故事

Ag 合金化 PbTe 存在一个有限寿命的高界面面积低热导率状态：PSD 和空间组织选择其粗化路径，而界面面积损失设定当前模型中的主要热导恢复。

### 冻结主 punchline

> Ag-alloyed PbTe exhibits a metastable low-thermal-conductivity window sustained by an interface-rich precipitate population. PSD and spatial organization determine how this population coarsens, whereas the loss of interfacial area determines how the thermal window closes.

中文含义是：Ag 合金化 PbTe 存在一个由高界面面积析出物人口维持的亚稳低热导率窗口。PSD 和空间组织决定这一人口如何粗化，而界面面积的损失决定这一热导窗口如何关闭。

最后一句必须理解为冻结 V2 界面散射合同内的机制结论。它不等于已经证明 `Sv` 唯一决定实验 total κ，也不等于当前模型能够预测完整 `zT`。

## 3. 论文真正回答的科学问题

本项目不把“平衡附近仍会继续粗化”本身当作创新。真正的问题是：在基体平均 Ag 浓度变化很小、而显微观测又不能唯一给出 6 h resolved-β 人口的条件下，实验允许的析出物人口能够发生多大程度的结构重排；PSD 与空间组织分别如何选择粗化路径；这种路径差异是否足以改变界面驱动的热导恢复。

主逻辑链冻结为：

Sheskin 6 h 低 κ 状态 → 不同显微技术的观测窗口不一致 → PF-resolved β 初态不可唯一识别 → 构建实验约束的允许微结构 ensemble → PSD 与空间实现选择粗化路径 → `Sv` 损失关闭模型中的低 κ 窗口 → 加工目标从复现唯一坐标转向延长高界面面积状态的寿命 → Yu 作为更广泛缺陷重分配的辅助机制证据。

## 4. Sheskin 与 Yu 的严格数据分工

冻结的数据边界句为：

> Sheskin and Yu are used as complementary but non-interchangeable datasets: the former defines the 6–48 h microstructure–thermal-conductivity trajectory, whereas the latter resolves the broader defect redistribution associated with long-term annealing.

Sheskin 2018 提供本文的 6→48 h 微观结构—热导率主轨迹，支持高密度 Ag-rich 纳米结构、后续粗化、界面密度降低和 κ 回升。它不能被写成完整的 6→48 h `zT` 轨迹，也不能与 Grossfeld–Sheskin 2017 静默合并。

Yu 2024 提供 AQ→48 h 端点上的 GP-like 对象、基体 Ag、β 析出物、应变松弛、Ag-decorated dislocations、电学和 `zT` 信息。它用于说明长期退火可能涉及多个缺陷储库之间的 Ag 重分配，但不能替 Sheskin 补充 6 h `zT`，也不能与 Sheskin 拼接为一条连续时效曲线。

## 5. 初态不可识别性与 ensemble 方法

PF-resolved stoichiometric β、APT-detected Ag-rich objects、GP-like/unresolved Ag-rich population 和 SEM/TEM-visible coarse precipitates 是四类不同对象。它们受到不同的组分阈值、空间分辨率、取样体积、2D/3D 几何和最小可见尺寸限制，不能直接一一对应。

因此，方法学声明不是“恢复真实 6 h seed population”，而是：

> We do not reconstruct a unique experimental 6 h precipitate population. Instead, we construct an ensemble of experimentally admissible resolved-β populations constrained by total Ag inventory, matrix composition and microscopy-dependent observation windows.

最小 observation operator 需要包含 blur、composition/phase threshold、connected components、minimum detectable size、resolution sensitivity、2D/3D sampling、watershed/neck separation，并输出可比较的 `Nv`、PSD、`Sv` 和 morphology。该算子尚未达到 production 状态，所以实验对象与 PF components 之间的定量一一验证仍是 P0。

## 6. 热力学与相场模型定位

相场部分是 globally conserved、coherent-elastic、post-nucleation propagation。它从实验允许的 6 h 状态向 48 h 传播，不预测绝对成核，也不声称唯一初态。

四个 solvus 点只支持 calibrated pseudo-binary thermodynamic description，不能称为 validated phase diagram。现有 21-case 轨迹使用 legacy runtime `L = 41212.9 - 18.05T`；独立 exact-fit candidate 已存在，但尚未成为 publication contract。最终论文必须选择一个热力学合同并完成影响审计或必要重算，不能混用 legacy 与 exact-fit 身份。

数值守恒、restart 和 elastic residual 只能称为 numerical qualification，不能替代实验验证。

## 7. Case 001–003 的条件性结果

三个 400³ cases 使用完全相同的 `REF_BROAD_METHOD1LIKE` 初始 PSD、512 个颗粒、总库存、热力学、物理和数值设置，只改变颗粒中心及半径—位置随机分配。三条轨迹均到达 48 h 完整 checkpoint。

| Case | 最终粒子数 | 平均半径 / nm | `Sv` / m⁻¹ | `M6` / nm³ | 573.15 K V2 MI recovery |
|---|---:|---:|---:|---:|---:|
| 001 | 21 | 24.5611 | 2.6327×10⁶ | 137.03 | 4.1005% |
| 002 | 19 | 25.4021 | 2.5397×10⁶ | 168.75 | 4.2075% |
| 003 | 21 | 24.9159 | 2.6654×10⁶ | 126.18 | 4.0718% |

三个 case 的 range/mean 分别为：平均半径 3.37%，`Sv` 4.81%，`M6` 29.56%，V2 Δκ 3.29%。因此，大尺寸尾部和颗粒存活对随机空间实现明显敏感，而当前界面驱动热响应在这三个 case 之间相对稳健。

冻结的核心写法为：

> At fixed initial PSD, precipitate inventory and thermodynamic conditions, random spatial realization substantially altered particle survival and the large-size tail of the evolving PSD, yet changed the predicted interface-driven thermal recovery by only approximately 3–4%. The coarsening pathway is therefore spatially sensitive, whereas the thermal response is comparatively robust.

这些数字的状态统一为 `COMPLETE_CHECKPOINT_CONDITIONAL_ANALYSIS`。旧 anchor 使用 64-particle schema 检查 512-particle fixture，merge-aware 统一后处理也尚未闭合；在修复并冻结复算前，不能称为 final production authority。A/B/C 的正式角色是 `fixed-PSD spatial uncertainty propagation and thermal-response robustness test`，不是 designed spatial control。

## 8. 三层机制分离

### PSD defines the size hierarchy

PSD 决定曲率竞争中的小颗粒供体、大颗粒受体及大尺寸尾部潜力。当前可以提出这个机制层级，但不同 PSD 家族的定量效应仍需完整 21-case campaign。

### Spatial organization selects the survival pathway

空间组织改变最近邻距离、局部捕获区、扩散竞争、弹性交互和 radius–position coupling，从而改变哪些颗粒存活以及 `M6` 如何演化。三个随机 realization 证明的是不确定性传播和结果稳健性，不是加工可控性。

### Interfacial-area loss sets the modeled thermal outcome

在冻结 V2 interface-scattering contract 中，`Sv` 损失控制热导恢复的方向和主要模型幅度，空间 realization 对该信号只产生次级调制。`M6` 是大尺寸尾部描述量，不是 κ 的唯一决定量。

核心概念是：

> Variables controlling the coarsening pathway are not identical to those controlling its thermal consequence.

这应作为 Discussion 的正面机制结论，而不是隐藏在 limitations 中。

## 9. 热输运结论与边界

当前模型研究 PF-informed resolved-particle/interface thermal transport，主要处理晶格侧的颗粒与界面散射杠杆。V1 density-contrast/full-PSD alone 不能产生实验所示的热导上升；冻结 V2 加入界面散射后得到稳定正号，并且 Case 001–003 的恢复幅度相近。

对于已测试的 full-PSD transport，`Sv+M6` 比 `Nv+Rmean` 或单独 `Sv` 更好地压缩 PSD 信息，但 full PSD 仍是 authority，因为矩闭合并不唯一。

模型没有完整计算 Seebeck coefficient、electrical conductivity、carrier concentration、weighted mobility、defect-dependent electronic transport、dislocation contribution，也没有严格拆分实验 total κ 中的 κe 与 κL。因此不能从当前结果直接预测完整 `zT`，也不能称为 high-`zT` window。

V2 捕捉了实验热导恢复的方向，但当前 resolved-particle/interface 模型的幅度不足以唯一解释更大的实验恢复。该残差的安全次级结论是：

> Resolved β coarsening governs the visible kinetic pathway, while the remaining transport discrepancy motivates contributions from unresolved Ag-rich populations and broader defect-state evolution.

不能利用残差反推某个唯一缺失机制，更不能宣称位错已经被证明导致 coherent-to-incoherent transition。

## 10. 论文结构与结尾方式

Acta 版本应以材料机制为主，而不是以工作流为主：Introduction 从 Sheskin 低 κ 状态和观测缺口开始；Methods 建立 thermodynamic/PF/ensemble/observation/transport 合同；Results 依次展示允许初态、粗化路径、Case 001–003 的结构—热响应差异，以及 full PSD 与 `Sv+M6`；Discussion 解释 pathway–outcome separation，再引入 Yu 的 broader defect redistribution；Conclusions 以窗口寿命和可测试加工方向收尾。

npj 版本可以强化 versioned ensemble、observation operator、uncertainty propagation 和 PF→transport handoff，但不能退化成只有软件工程价值的 workflow paper。材料 punchline 必须仍然处于中心。

结尾不应是“模型不能完整解释实验”。推荐的最终落点是：

> The processing target is not a unique precipitate arrangement, but the lifetime of the interface-rich low-κ state.

温度历史和 elastic bias 只能写成 testable processing routes。尚不能写成它们已经稳定窗口、控制空间分布或构成已验证的加工策略。

## 11. 八张主图的叙事任务

1. Sheskin 低 κ 状态、6→48 h 轨迹与 observation gap。
2. Thermodynamic/PF/ensemble/transport 框架及数据角色。
3. 实验允许的初态与 fixed-PSD spatial replicates。
4. 代表性 6→48 h 粗化过程。
5. PSD 和空间 realization 对 survival、`Sv`、`M6` 的影响。
6. Case 001–003：pathway-sensitive but transport-robust，是当前决定性主图。
7. Full PSD 与 `Nv+Rmean`、`Sv`、`M6`、`Sv+M6` 的压缩比较。
8. 低 κ 窗口关闭、与实验的有界比较及 Yu 机制背景。

在 21 cases 和 authority gates 未完成前，Figures 2–6 中涉及精确 400³ 数字或 factorial effect 的面板必须标为 `CONDITIONAL` 或 `BLOCKED`。

## 12. 与 Acta 和 npj 论文的客观比较

当前 idea 在概念上与强 Acta/npj 论文 comparable，因为它不是报告普通粗化，而是提出并量化“路径选择变量”与“热学结果变量”并不相同。它与 Acta 范例中的 post-nucleation handoff、bounded mechanism inference、observation/data-role discipline 和 spatial causality 具有对应关系；与 npj 范例中的 versioned contract 和 uncertainty-propagation workflow 也有对应关系。

它目前弱于最强范例的地方是：observation operator 未实施；400³ 仍为 conditional authority；只有三个随机 realization；没有 designed spatial architecture；thermodynamic 和 thermal quantity identity 未冻结；没有完整 21-case 因子统计、forward processing loop、transfer test 和公开 archive。

因此，punchline 已有资格闭合论文的科学叙事和 Discussion/Conclusions，但尚不足以闭合投稿证据链。当前评分为：scientific story maturity 7.8/10，punchline conceptual novelty 7.5/10，evidence closure 4.8/10，Acta readiness 4.5/10，npj readiness 5.0/10。

保持 Acta-oriented 架构是合理的长期目标。若不增加 designed/experimental control，在最低 P0 闭合后，Computational Materials Science 是更现实的投稿路径；若机制分析更紧凑，可考虑 Physical Review Materials。npj 需要把 operator、scalability、transfer 和 archive 提升为一等证据。

## 13. P0、P1 与执行顺序

### P0：投稿前必须闭合

1. 修复 512-particle V2 fixture 与旧 64-particle schema 冲突。
2. 冻结 merge-aware 统一后处理及 object/lineage definitions。
3. 完成 Case 001–003 production-authority 复算并更新所有百分比。
4. 完成全部 21 cases；在此之前不报告 factorial PSD/density/space effects。
5. 冻结一个 publication thermodynamic contract，并审计或重算中心结论。
6. 统一 thermal quantity identity，明确 modeled lattice-style response 与 measured total κ 的关系。
7. 实现最小 observation operator 及其 sensitivity。

### P1：支撑最强 Acta 机制与加工主张

- 完成 matched finite-size bridge；
- 运行 clustered、uniform、large-particle-separated、size–position-correlated 和 strain-biased architectures；
- 建立 local spatial descriptors 与 particle fate 的统计关联；
- 完成 thermal-history/temperature 与 elastic-bias proof-of-concept；
- 设计能部分解耦 `Sv`、`M6` 和空间结构的对照。

推荐顺序为：schema repair → merge-aware freeze → Case 001–003 authority → thermodynamic decision → thermal identity → observation operator → 21 cases → finite-size bridge → designed spatial controls → temperature/elastic tests → publication archive。

## 14. 当前可以写与暂时不能写的内容

现在可以开始写 Introduction、Methods 的数据角色与模型合同、Discussion 的三层机制，以及 Conclusions 的 window-lifetime implication。Case 001–003 的精确数字可放入内部草稿，但必须保留 `CONDITIONAL` 标记，等待 authority rerun 后再冻结正文和图注。

当前不能声明：恢复了真实 6 h 初态；Sheskin 给出完整 `zT` 轨迹；6 h 是已证明的最高 `zT` 状态；Yu 与 Sheskin 是同一轨迹；三个随机 seeds 证明空间可控；空间对所有热输运都不重要；温度或应变已稳定低 κ 窗口；resolved β 单独定量解释实验 total κ；Case 001–003 已是 final production authority；21-case campaign 已证明加工可控性。

## 15. V3 写作库使用入口

- 完整科学故事：`01_frozen_story_v3_metastable_low_k_window.md`
- Acta 与 npj 结构：`02_acta_oriented_manuscript_blueprint_v3.md`、`03_npj_oriented_manuscript_blueprint_v3.md`
- 摘要与标题：`06_abstract_blueprints_v3.md`、`07_title_strategy_v3.md`
- Claim 与证据：`08_claim_evidence_language_guide_v3.md`、`claim_evidence_matrix_v3.csv`
- Observation 与空间因果：`10_observation_operator_writing_plan_v3.md`、`12_spatial_causality_writing_plan_v3.md`
- 图与主文/SI 边界：`13_figure_narrative_v3.md`、`14_main_text_si_boundary_v3.md`
- 禁止过度主张与审稿风险：`16_forbidden_overclaims_v3.md`、`17_reviewer_risk_matrix_v3.csv`
- 证据优先级与期刊路线：`18_missing_evidence_priority_v3.md`、`19_recommended_target_journal_v3.md`
- 完整验收结论：`frozen_story_v3_acceptance_report.md`
- 文件版本、hash 与来源：`updated_writing_library_manifest.csv`、`writing_library_reframe_inventory.md`、`evidence_snapshot_provenance_v3.md`

在后续写作中，以本文件和 `01_frozen_story_v3_metastable_low_k_window.md` 作为故事入口，以 `claim_evidence_matrix_v3.csv` 判断每个句子的证据等级；当 production-authority 结果变化时，先更新 evidence snapshots 和 claim matrix，再更新摘要、图注和精确数值。
