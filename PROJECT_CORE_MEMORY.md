# 项目核心记忆

> 本文件是 `CUDA_STO_PF` 当前课题范围、科学边界、数值合同、路线图和
> 当前状态的权威入口。任何代理在修改代码、设计算例、解释报告或提交
> 生产任务前，都必须先读取本文件。
>
> 若历史报告与本文件冲突，以本文件中的研究边界和最新状态更新为准。
> 历史结果不得删除，但必须按其原始来源、哈希、物理开关和证据等级解释。

# 项目总纲

## 一、课题定位

暂定题目：

> PbTe–Ag₂Te后成核微观结构条件演化路径及其晶格热输运响应

核心问题：

- 给定6 h实验约束，不同初始粒径分布和空间结构如何演化到48 h？
- 哪些初始结构能够同时解释颗粒粗化和基体Ag浓度近似稳定？
- 完整PSD、数密度、平均半径、\(S_v\)、\(M_6\) 中，哪些是预测晶格热导率所必需的描述量？

## 二、研究边界

本阶段包含：

- 已有resolved β颗粒的溶解、长大和Ostwald粗化；
- 守恒相场演化；
- 粒径分布和空间结构不确定性；
- 6–48 h实验趋势比较；
- PF结果到晶格热导率模型的连接。

本阶段不包含：

- 绝对首次β成核时间预测；
- GP辅助成核；
- GP Birth和GP release；
- 为匹配结果进行案例专用参数调整；
- 超出当前固定胞/周期边界、单取向eigenstrain合同的弹性扩展；
- 位错管道扩散和非相干界面机制。

## 三、实验数据与物理约束

### 3.1 主要实验锚点

380 °C下：

- 6 h基体Ag：\(0.62\pm0.04\) at.%；
- 48 h基体Ag：\(0.62\pm0.04\) at.%；
- 析出对象数密度显著下降；
- 析出物尺寸增大；
- 6 h附近为成核主导向粗化主导转变的近似接管点。

### 3.2 辅助实验信息

整理：

- 6 h和48 h粒径分布；
- 形貌、长宽比和取向关系；
- Ag-rich对象与化学计量Ag₂Te的区别；
- 数密度、体积分数和总Ag库存；
- 界面能、晶格失配和位错密度；
- 热导率及Debye–Callaway参数。

### 3.3 实验到PF的映射合同

明确区分：

- APT Ag-rich对象密度；
- TEM可见析出物密度；
- PF resolved β连通域密度；
- unresolved Ag-rich库存；
- resolved Ag₂Te库存。

## 四、数值引擎

### 4.1 相场变量

主要演化：

- β相场 \(\phi\)；
- 守恒组分变量 \(Y/x_B^\alpha\)；
- 守恒零模；
- 历史时间层 \(dY/dt_{\rm prev}\)。

### 4.2 数值方法

冻结：

- \(dx=1\) nm；
- \(\lambda_{\rm sm}=4\) nm；
- 显式空间演化；
- Ji-Chen守恒零模；
- checkpoint/restart零模provenance；
- 无GP、无外部源、无新成核基线。

### 4.3 数值验收

验证：

- dt收敛；
- 质量守恒；
- 零模残差；
- continuous/restart字节一致；
- 颗粒跟踪稳定；
- 周期边界正确；
- 盒子尺寸敏感性；
- GPU性能与显存。

## 五、6 h初始状态构造

### 5.1 总库存合同

固定：

- 全局Ag/B库存；
- 基体浓度；
- resolved β库存；
- 必要时单独记录unresolved库存。

### 5.2 初始PSD族

计划建立：

- 窄连续PSD；
- 宽连续PSD；
- 双峰PSD；
- 实验直接约束PSD；
- 不同随机空间排布。

### 5.3 单粒子profile库

建立不同半径下的：

- \(\phi(r)\)；
- \(x_B(r)\)；
- 局部化学势；
- 界面profile；
- 有效h体积；
- 质量库存。

目标是避免解析tanh球产生非物理初始松弛。

### 5.4 空间排布

采用：

- 周期Poisson-disk或硬核随机位置；
- 无规则晶格；
- 无周期镜像重叠；
- 多个hash-pinned随机种子。

## 六、初态筛选阶段

每个候选初态先运行6–12 h短预检，检查：

- 是否存在初始profile突跳；
- β总体积是增长、下降还是近似稳定；
- 小颗粒溶解与大颗粒增长是否平衡；
- 远场基体浓度轨迹；
- 质量、零模、dt和restart；
- 是否出现非预期合并或分裂。

这一阶段用于排除数值不自洽初态，不等同于12 h实验验证，因为目前没有12 h实验数据。

## 七、正式6–48 h条件路径模拟

对通过预检的候选运行：

- 6、12、18、24、36、48 h输出；
- 多个PSD；
- 多个空间随机种子；
- 相同总库存和物理参数。

主要输出：

- \(N_v(t)\)；
- \(R_{\rm mean}(t)\)；
- 完整PSD；
- β体积分数；
- \(S_v(t)\)；
- \(M_6(t)\)；
- 基体远场Ag；
- 单粒子轨迹；
- 溶解事件；
- 空间相关统计。

## 八、实验观察算子

分别建立：

- PF真实连通域统计；
- APT-like ROI；
- TEM-like尺寸阈值；
- 远场基体浓度；
- 周期距离排除耗尽层；
- 不同检测阈值下的可见颗粒密度。

避免直接把PF颗粒数与APT对象数一对一比较。

## 九、48 h实验验证

比较：

- 数密度下降方向和幅度；
- 平均尺寸增长；
- PSD展宽或收缩；
- β体积分数变化；
- 基体Ag是否仍落在实验区间；
- 形貌和界面面积变化。

输出不是唯一“拟合曲线”，而是实验允许的条件路径集合及其不确定性。

## 十、晶格热输运模块

### 10.1 Yu模型复现

先独立复现：

- AQ状态晶格热导率；
- 48 h状态晶格热导率；
- Debye–Callaway基线；
- 点缺陷、Umklapp、边界、位错和析出物散射。

### 10.2 PF到输运的接口

从PF提取：

- \(N_v\)；
- PSD；
- \(S_v\)；
- \(M_6=\int R^6n(R)\,dR\)；
- 形貌和取向统计。

### 10.3 描述量充分性测试

依次比较：

- \(N_v+\bar R\)；
- \(N_v+\bar R+CV\)；
- \(S_v\)；
- \(S_v+M_6\)；
- 完整PSD积分。

确定预测 \(\kappa_L\) 所需的最小微观结构描述量。

## 十一、扩展物理

在无弹性基线通过后，再依次研究：

- 弹性应变；
- 取向相关界面能；
- 非球形颗粒；
- 位错管道扩散；
- unresolved与resolved双尺度库存；
- 温度扩展到400和450 °C。

每项扩展都与无弹性基线单独比较。

## 十二、最终成果

预期形成：

1. 守恒PF后成核演化引擎；
2. 实验约束的6 h初态生成方法；
3. 多条6–48 h条件粗化路径；
4. PF/APT/TEM统一观察算子；
5. 微观结构到晶格热导率的耦合模型；
6. 最小充分微观结构描述量；
7. 模型适用边界和不确定性说明。

## 当前进度（原始总纲记录）

已经完成：

- 守恒PF和零模数值引擎；
- dt与restart验收；
- 随机连续PSD V2初态；
- 6–12 h异质颗粒竞争；
- 周期远场和APT-like观察算子。

当时记录的问题：

- V2在12 h出现真实的基体Ag下降；
- 当前初态包含未平衡解析profile；
- PSD较窄；
- resolved β与实验Ag-rich总库存映射尚未完全闭合。

该历史节点为：

```text
数值引擎已验收
→ 观察算子已验收
→ 6 h初态科学闭合待决策
→ 尚未进入正式6–48 h生产模拟
```

上面的节点是总纲形成时的历史状态，不是当前最新状态。当前状态以下一节为准。

## 十三、2026-07-30最新核心状态

### 13.1 已完成的有弹性6–48 h路径

已完成匹配初态、匹配网格、匹配物理参数的246³、T380、有弹性
6–48 h运行及43个注册快照审计。

核心端点：

- resolved颗粒数：96 → 6；
- 平均h体积等效半径：9.5405 → 23.5174 nm；
- β体积分数：0.0239295 → 0.0237833，变化约−0.61%；
- \(S_v\)：0.0074253 → 0.0028766 nm⁻¹，下降约61.3%；
- \(M_6\)：5.3626 → 100.0022 nm³，增加约18.65倍；
- 基体 \(x_{\rm Ag}\)：0.6201 → 0.6348 at.%；
- 最终基体Ag仍位于实验 \(0.62\pm0.04\) at.%区间；
- 质量、零模、checkpoint/restart和总体PSD闭合通过；
- GP、GP Birth、GP release和新β成核均关闭。

### 13.2 当前Ostwald判定

当前数据强烈支持弹性修正的、粗化主导的Ostwald演化：

- 小颗粒优先消失；
- 初始最小四分位颗粒到8 h全部消失；
- 初始最大四分位颗粒到8 h全部存活；
- \(\langle R^3\rangle\) 在8–48 h近似线性增长，\(R^2\approx0.967\)；
- \(1/N\) 对时间近似线性，\(R^2\approx0.966\)；
- \(N\langle R^3\rangle\) 与β体积分数整体近似守恒；
- 界面面积持续下降；
- 没有新resolved颗粒产生。

当前不能宣称纯经典LSW，因为：

- 6–8 h存在明显的弹性初态重构；
- 8–48 h仍包含少量净β生长；
- 15–17 h存在已确认的diffuse-neck合并谱系。

### 13.3 增量机制审计已闭合

原有三个正式阻塞项已经通过增量方式闭合，不需要重跑整套6–48 h：

```text
PASS_INCREMENTAL_MERGE_MORPHOLOGY_AND_RUNTIME_ELASTIC_SCALAR_V1
```

具体证据：

- 15–16 h异常已确认为颗粒46与68的2→1连通；
- 16 h只在 `h<=0.001` 连通，17 h才形成 `h>=0.005` 的较强物质颈部，
  因此登记为diffuse-neck onset及持续merge-lineage，不再误计为颗粒68溶解；
- 6–8 h完成30帧高时间分辨形貌审计，无意外merge/split；
- 运行时已输出非零平均/峰值弹性能和静水应力范围；
- 新诊断运行的8 h `phi`、`xB`、`xBtot` 与原轨迹逐字节一致；
- Ji-Chen守恒零模通过，最终平均质量误差约 \(2.85\times10^{-16}\)。

据此，当前单盒路径登记为：

```text
PASS_MIXED_OSTWALD_AND_COALESCENCE_COARSENING_V1
```

它是粗化主导、包含少量物理合并的弹性条件路径，不是纯经典LSW。
旧48 h轨迹无需重跑；以后新生产运行自然携带完整弹性能标量。

### 13.4 有限尺寸与统计边界

48 h最终只有6个resolved颗粒，其中两个28–30 nm颗粒贡献约76.5%的
\(M_6\)。因此单一246³盒子可证明一条条件演化路径，但不足以给出稳健
的高阶矩或热输运统计。

下一阶段必须采用：

- 至少3个hash-pinned随机PSD重复；或
- 一个更大物理盒子；
- 报告 \(N_v\)、PSD、\(S_v\)、\(M_6\)、基体Ag的均值、离散度和有限尺寸敏感性。

### 13.5 晶格热输运与位错边界

当前PF没有位错密度演化变量，也不包含塑性、位错成核、滑移或位错管道
扩散。弹性场不等于位错场，不得从析出物密度、弹性能或eigenstrain
反推出 \(N_D\)。

PF可以向输运模型提供：

- \(N_v(t)\)；
- 完整PSD；
- \(S_v(t)\)；
- \(M_6(t)\)；
- 基体Ag；
- 形貌与取向。

位错散射必须作为外部条件：

```text
DISLOCATION_OFF
DISLOCATION_FIXED_EXTERNAL
DISLOCATION_SENSITIVITY_BAND
```

Yu 2024复现的当前边界：

- AQ公开模型复现通过；
- 48 h严格使用公开S13和表S2参数时不通过；
- 公开48 h曲线对应约0.119倍的等效位错散射强度；
- 0.119只能作为反演诊断，不能宣称作者拟合了位错密度；
- Yu样品的位错密度不得直接移植到本项目样品。

### 13.6 当前路线

```text
弹性约束target-profile库（已验收）
→ 守恒多粒子6 h fixture materializer
→ 多粒子fixture的dt/restart/身份/能量验收
→ 多随机初态/更大盒子统计
→ PF完整PSD接入晶格热输运
→ 无位错/外部位错条件路径
→ 6–48 h热导率趋势和实验对比
```

### 13.7 当前完成度

以下百分比是项目管理估算，不是物理验收标记：

| 目标 | 当前完成度 |
|---|---:|
| PF数值引擎、守恒、重启 | 约95% |
| 有弹性6–48 h粗化机制 | 约90% |
| 弹性单粒子target-profile初态库 | 100%（V1注册范围） |
| 实验浓度与微观结构趋势 | 约80% |
| 多随机样本统计资格 | 约20% |
| PF到晶格热导率耦合 | 约35% |
| 完整“微观结构→热电性能”课题 | 约65% |

当前主要未完成项是：

- 从已验收单粒子库构造守恒、无重叠、多粒子6 h初态；
- 多随机初态或更大盒子的统计资格；
- 完整PSD到晶格热输运的条件路径。

### 13.8 弹性约束target-profile库V1

已完成380 °C、\(dx=1\) nm、\(\lambda_{\rm sm}=4\) nm的离线弹性
约束松弛库。固定合同为周期固定胞、identity/[100]取向、
eigenstrain \((0.046,-0.022,-0.017,0,0,0)\)，外加应变和外加应力为0。
GP、外部源和新β成核全部关闭。

注册半径：

```text
8.0,8.5,9.0,9.5,10.0,10.5,11.0,11.5 nm
```

最终选择的是cluster `minimize_dt=0.025`库：

```text
library_status=PASS_PF_ELASTIC_TARGET_PROFILE_LIBRARY_V1
profile_count=8
library_manifest_sha256=58803a8bc6679b823e45e7a7b85df16ae68efa55338d52d4c4151b414a5ef0fe
selection_provenance_sha256=56c44d8f72b27bb462dffe89b59cb2fcb2ff0bf8807dec9d9cac31d2bd7fcbe3
source_tree_sha256=f7855699addf98f9d5aed03af876d851d524fb62c98f5a48df78d1fe561a2a75
binary_sha256=55cf917df94fcf01373d62f54f9ab99715975863ad5dcfb95baa8a517460cda1
```

已通过：

- 8/8约束体积、exact canonical mass、KKT和收敛审计；
- 弹性能量项及eigenstrain provenance；
- cluster `dt=0.05` 对 `dt=0.025` 细化；
- 96³、128³、160³、192³有限盒子审计；
- R=8.0、9.5、11.5 nm的动态接管、dt、零模、单粒子身份；
- 三个动态案例的continuous/restart checkpoint逐字节一致。

最终工程状态：

```text
PASS_PF_ELASTIC_TARGET_PROFILE_LIBRARY_ENGINEERING_V1
```

强制边界：

- 只有8个精确注册半径可直接使用；
- 未验收半径插值、旋转取向和多粒子场叠加；
- raw `xB_alpha`只允许相同孤立网格直接加载；
- 跨盒子可移植量是
  `delta_C_relaxation=(1-h)(xB-xB_far)`；
- 多粒子目标盒必须重新构造matrix baseline，并用守恒零模关闭精确总库存；
- 本库不是实验6 h PSD，也不是6–48 h生产结果。

### 13.9 多粒子目标远场重组测试

已验证“保持单粒子库的弹性 \(\phi\) 与局部
`delta_x_alpha` 修正，只按目标远场浓度重组多粒子盒”的方案。
六粒子96³验证fixture固定
\(x_{B,\mathrm{matrix}}=0.006219279767278563\)
（\(x_{\rm Ag}=0.0062\)），并让全局库存自然导出为
\(\langle C_B\rangle=0.029996585491183673\)，而不是强制0.03。

静态守恒、6/6颗粒身份、零模和continuous/restart逐字节一致均通过。
但是相同物理终点的 `dt=0.02` 对 `dt=0.01` 仍得到
\(x_B\) MAE \(1.1170\times10^{-4}\)，高于 \(5\times10^{-5}\)
硬门槛；旧强制0.03方案为 \(1.1145\times10^{-4}\)，两者实质相同。

因此：

```text
BLOCKED_MULTI_PARTICLE_COMMON_EQUILIBRIUM_NOT_RESTORED_BY_FAR_FIELD_REBASE
```

远场/总库存之间约 \(3.4\times10^{-6}\) 的差异不是dt失败原因。
多个独立单粒子平衡场进入同一周期固定胞后，非局部弹性相互作用使其
不再是多粒子共同平衡态。后续不得继续通过调整统一远场基线解决该问题；
若坚持零初始瞬态，应开发真正收敛的逐粒子体积约束优化器，并先在同一
六粒子验证fixture上闭合KKT、dt和restart。

### 13.10 正式条件接管初态策略V1

2026-07-31正式冻结并验收：

```text
MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1
```

本策略将6 h多颗粒初态的科学准入要求限定为：

1. 每个resolved β颗粒逐一来自13.8节冻结的、精确注册半径的
   hash-pinned弹性target-profile库；
2. 目标周期盒的全局Ag/B canonical inventory在机器精度下闭合。

13.9节的共同多颗粒平衡失败仍是有效历史诊断，但从本节起，共同化学—
弹性平衡不再是“条件初态”的准入门槛。不得因此宣称初态已经共同平衡，
也不得把它称为实验唯一的6 h微观结构。

冻结的六粒子验证fixture为96³，使用两个10.5 nm、两个9.5 nm和两个
8.0 nm的identity/[100]库profile。正式身份：

```text
profile_library_manifest_sha256=58803a8bc6679b823e45e7a7b85df16ae68efa55338d52d4c4151b414a5ef0fe
conditional_fixture_manifest_sha256=a87e76405bd1b901b6848b6c39d788ff3fe537ca0acea93f99b64ac67e6ced81
initial_inventory_relative_error=2.7412914188275474e-16
```

实现和验收结果：

- 每个颗粒的库entry、半径、中心、取向、profile manifest、`phi`、
  `delta_C_relaxation`、有效h体积和canonical inventory均已记录；
- 跨盒子组合不复制孤立盒子的absolute `xB_alpha`，而使用冻结的
  `delta_C_relaxation`并在目标盒反算matrix baseline；
- manifest顺序不变性和重复材料化字节确定性通过；
- 18项schema、负例、质量、零模、自由首步和禁止路径测试全部通过；
- checkpoint V3保存并验证initial-state class、fixture SHA-256和
  profile-library SHA-256；
- 64步连续运行与32+32步restart的完整checkpoint逐字节一致；
- 错误fixture/library provenance均在恢复前fail closed；
- 第一个动态步后profile能够自由变化，没有锁定；
- GP、GP Birth、GP release、外部源和新β成核全部关闭；
- 未调用多颗粒共同平衡minimizer或逐颗粒体积乘子优化器。

dt政策同时正式修改：

```text
full_field_xB_dt_MAE_blocking=false
```

`dt=0.02`与`dt=0.01`在相同终点的full-field
\(x_B\) MAE仍为 \(1.1145\times10^{-4}\)，必须保留并用于生产dt选择和
时间离散不确定性报告，但不再否决fixture的库映射/质量守恒身份。生产dt
仍必须另行通过数值稳定性和主要物理观察量收敛审计。

时间语义冻结为：

```text
common_multi_particle_equilibrium_required=false
common_multi_particle_equilibrium_claim=false
initial_relaxation_is_physical_evolution=true
particle_profiles_locked_after_t0=false
```

不运行预松弛；完整周期盒中的弹性重构从6 h开始即计入真实物理时间，
不得丢弃后重新标记为6 h。

最终状态：

```text
PASS_MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1
```

本状态只验收初态生成、质量合同、provenance和短restart功能，尚未授权或
启动新的6–48 h正式多初态生产。

### 13.11 主要物理观察量生产dt选择V1

2026-07-31在13.10节冻结的六粒子96³条件接管fixture上完成三档共同终点
收敛审计：

```text
dt_code=0.02,  256 steps, dt_physical=0.9909260953431841 s
dt_code=0.01,  512 steps, dt_physical=0.4954630476715921 s
dt_code=0.005, 1024 steps, dt_physical=0.2477315238357960 s
common_code_time=5.12
common_physical_time=253.67708040785513 s
```

以`dt=0.005`为参考，`dt=0.02`的主要误差为：

- β体积分数相对误差 \(5.2579\times10^{-4}\)；
- 平均等效半径相对误差 \(2.3022\times10^{-4}\)；
- \(S_v\)相对误差 \(4.0343\times10^{-4}\)；
- \(M_6\)相对误差 \(6.6382\times10^{-4}\)；
- matrix \(x_{\rm Ag}\)绝对差 \(1.1908\times10^{-5}\)；
- 平均弹性能相对误差 \(6.5185\times10^{-4}\)；
- canonical inventory相对误差 \(1.6050\times10^{-13}\)。

三档均保持6个颗粒及精确身份映射，无merge/split，误差随dt细化单调
下降，stderr为空，守恒零模和禁止路径审计通过。全场\(x_B\) MAE继续作为
非阻塞时间离散诊断量。

分析器边界合同已与CUDA求解器统一：
\(\phi\in[-10^{-6},1+10^{-6}]\)。这只修正审计器原先错误使用严格
\([0,1]\)造成的共同假阻塞，没有修改求解器、物理参数或运行场。

`dt=0.02`连续256步与128+restart+128步的完整checkpoint逐字节一致，
共同SHA-256为：

```text
148bd979e09047907f7b30f1f8904b5569a4f201d556750293f6f8f629085d9c
```

正式选择：

```text
selected_production_dt_code=0.02
selected_production_dt_physical_s=0.9909260953431841
fallback_dt_code=0.01
final_status=PASS_PRODUCTION_DT_OBSERVABLE_CONVERGENCE_V1
```

关键provenance：

```text
workstation_root=/home/zhiheng/tmp/pf_production_dt_observable_selection_v2_20260731
fixture_manifest_sha256=a87e76405bd1b901b6848b6c39d788ff3fe537ca0acea93f99b64ac67e6ced81
profile_library_manifest_sha256=58803a8bc6679b823e45e7a7b85df16ae68efa55338d52d4c4151b414a5ef0fe
binary_sha256=7581c169fb1d1c16ee60764f418ee9c33508682c8b9dd8cf76b89368c7990602
analysis_script_sha256=2eacc7b836c4f4ab463b752c26e9d3f972afda1ba5c75611f103c2f0063dfc8d
final_audit_sha256=234fece90c5a0b94a252ed6f6af6b07c9c611ef9f0c08f2b8765ded327fccede
```

本节只冻结当前380 °C条件接管路径的生产dt。尚未运行新的6–48 h正式
多初态生产，也没有将该dt外推验收到其他温度、网格或物理扩展。

### 13.12 三个246³ hash-pinned条件接管fixture短资格V1

2026-07-31完成三个独立空间重复、每个96颗粒的生产规模fixture构造、
静态资格、256步连续/重启和6–8 h短筛选：

```text
PASS_246CUBE_THREE_SEED_LIBRARY_HANDOFF_SHORT_QUALIFICATION_V1
```

共同合同：

```text
domain=246x246x246
temperature_C=380
dx_nm=1
lambda_sm_nm=4
particle_count_per_fixture=96
initial_state_class=MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1
target_mean_C_Btot=0.03
dt_code=0.02
dt_physical_s=0.9909260953431841
elasticity_enabled=true
GP_enabled=false
GP_birth_enabled=false
GP_release_enabled=false
new_beta_nucleation_enabled=false
external_source_enabled=false
common_multi_particle_equilibrium_required=false
initial_relaxation_is_physical_evolution=true
```

冻结的离散PSD对三个样本完全相同：

```text
R=8.0 nm: 4
R=8.5 nm: 19
R=9.0 nm: 19
R=9.5 nm: 17
R=10.0 nm: 17
R=10.5 nm: 16
R=11.0 nm: 4
R=11.5 nm: 0
```

离散化后 \(\langle R^3\rangle\) 相对历史连续PSD变化为
\(+0.0011415672423844539\)。未使用profile插值、缩放、解析球替代、
旋转或多颗粒minimizer。

三个fixture身份：

| replicate | seed | fixture manifest SHA-256 | 反算matrix \(x_B\) | 对应 \(x_{\rm Ag}\) |
|---|---:|---|---:|---:|
| A | 18278234711707939752 | `63a5080b01962bf19f72a37541ffde4302fec3a0e4dc9e519b0f759f30367de6` | 0.006810401095358129 | 0.006787289015086700 |
| B | 6256128897973916905 | `b37682e5aea7cc294a675ce562a34fb0d306181990d40090df1d20779a93980c` | 0.006810340069425079 | 0.006787228402649627 |
| C | 4209997954605651191 | `12c265be4e352392385e689c87ecaea1d18dc364428c111fab5bceb1eac0f981` | 0.006810406213610769 | 0.006787294098659213 |

这些matrix值是固定全局
\(\langle C_B^{tot}\rangle=0.03\)、保留96个真实库profile及其局部
relaxation inventory后反算的守恒余量，不是强制的实验
\(x_{\rm Ag}=0.0062\)。matrix浓度在本资格中是科学诊断，不是fixture
身份否决项。

静态结果：

- 三个中心集合独立、非规则晶格、周期硬核分离通过；
- 288/288颗粒逐一映射到冻结库精确entry；
- manifest逆序材料化与重复材料化原始场逐字节一致；
- 三个目标盒的canonical inventory均在机器精度闭合；
- 三个fixture的中心、`phi`和`xB`哈希互不相同；
- GP、外部源、新β成核及多颗粒KKT/minimizer路径全部关闭。

256步资格：

```text
continuous=256
restart=128+128
endpoint_age_h=6.070465855668848
restart_status=PASS_BYTEWISE_ALL_THREE
```

三个连续和重启最终checkpoint逐字节一致，完整覆盖`phi`、`Y`、`xB`、
`dY_dt_prev`、零模、弹性运行时状态、ledger和checkpoint provenance。

6–8 h固定端点为：

```text
steps=7266
actual_endpoint_age_h=8.000019169100993
endpoint_time_error_s=+0.0690087636
```

短筛选结果：

| replicate | 8 h连通域数 | 合格溶解 | 已解析merge | β体积分数 | 平均R (nm) | \(S_v\) (nm⁻¹) | \(M_6\) (nm³) | 远场 \(x_{\rm Ag}\) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 60 | 36 | 0 | 0.0227586238629 | 10.6254419417 | 0.00596732871612 | 9.2129133545 | 0.00738837132637 |
| B | 54 | 42 | 0 | 0.0228309240198 | 11.1310448394 | 0.00582104678046 | 9.95186793732 | 0.00731452799925 |
| C | 58 | 37 | 1 | 0.0227691596308 | 10.6359822479 | 0.00586483726352 | 9.69260431821 | 0.00737718442361 |

C的严格`h>1e-4`追踪器最初在step 6656 fail-closed检测到2→1连接。
原始BLOCKED状态和事件表均已保留。只读补充审计建立持续组：

```text
merge_group_id=MG_P063_P086
merge_step=6656
merge_age_h=7.832112247390064
parent_support_overlap_fraction=0.99984661400414143,1.0
h_gt_1e-4_merge=true
h_gt_1e-3_merge=false
h_gt_5e-3_merge=false
neck_classification=DIFFUSE_TAIL_NECK_ONLY
```

该事件未被重标为dissolution，也没有通过改变主追踪阈值消除；它以显式
parent/child边和persistent merge-lineage group继续。三个阈值均无split、
新component、弱父映射或未合格溶解，最终补充状态为：

```text
PASS_246CUBE_RESOLVED_MERGE_AWARE_LINEAGE_V1
```

最终V2只读审计的分析脚本、tracker源码/二进制、fixture、初始component、
step 7266 checkpoint、审计JSON和两个权威PASS状态的SHA-256均冻结于：

```text
reports/pf_246cube_three_seed_library_handoff_v1/
merge_aware_C_analysis_provenance.sha256
```

工程与守恒结果：

```text
max_system_mass_drift=3.5033444208031475e-13
max_zero_mode_lambda=1.5678693265318445e-06
workstation_A_wall_s_per_step=1.254923680456491
cluster_B_wall_s_per_step=0.42146847360912976
cluster_C_wall_s_per_step=0.42160114122681874
peak_VRAM_MiB_A=4681
peak_VRAM_MiB_BC=4411
particle_lineage_status=PASS_ALL_THREE_RESOLVED
```

完整证据位于：

```text
reports/pf_246cube_three_seed_library_handoff_v1/
```

其中包含三个fixture manifest、PSD、空间统计、质量、restart、三组完整
观察量和P000–P095谱系、C的原始严格阻塞与merge-aware补充证据、性能、
最终决策及逐文件SHA-256 manifest。

本状态只说明三个fixture已经具备未来完整6–48 h ensemble的数值和工程
资格。它不声称三个初态是唯一实验6 h微观结构，也不声称共同多颗粒平衡。
截至13.12所记录的资格阶段，尚未启动完整6–48 h任务；后续启动状态以
紧接其后的13.13为准。

### 13.13 A/B/C完整6–48 h生产已启动（2026-07-31）

用户已经明确授权三个条件路径的完整生产。A/B/C全部从各自原始、哈希
固定的6 h raw fixture开始，采用共同的cluster `sm_80`源码和二进制：

```text
dt_code=0.02
dt_physical_s=0.9909260953431841
final_step=152585
science_steps=0,21798,43596,65393,108989,152585
checkpoint_cadence_steps=3633
checkpoint_cadence_physical_s=3600.034504381788
checkpoint_count_per_path=44
```

GP、GP Birth、GP release、外部源和新β成核继续关闭。12 h浓度仅做
只读观察，不自动终止条件路径。多阈值merge-aware谱系将在完整检查点链
上统一分析。

三条输入均通过fixture manifest、`phi/xB`、初始component/particle清单、
总库存和禁用路径哈希预检。Slurm登记为：

```text
A=73230 RUNNING gpu1 NVIDIA A100-SXM4-40GB
B=73231 PENDING QOSMaxJobsPerUserLimit
C=73232 PENDING QOSMaxJobsPerUserLimit
```

非覆盖输出根：

```text
/data/home/luozhiheng/tmp/pf_246cube_6h48h_A_v1_20260731
/data/home/luozhiheng/tmp/pf_246cube_6h48h_B_v1_20260731
/data/home/luozhiheng/tmp/pf_246cube_6h48h_C_v1_20260731
```

启动时A的stderr为空、GPU采样利用率约99–100%、显存约4.4 GiB。
`gpu_uvip`当前QOS限制该用户同时只运行一个作业，因此三条已全部排队，
但将自动串行接续。完成后统一计算A/B/C均值、标准差和范围，并按实验
允许条件路径或科学排除初态族分级；不得通过物理参数调整改变分级。

用户随后明确授权两个冗余、非覆盖副本，且保留原B/C排队任务：

```text
C gpu_vip_24h copy=73235
C vip root=/data/home/luozhiheng/tmp/pf_246cube_6h48h_C_vip24h_v1_20260731
C vip launch state=PENDING_PRIORITY
B workstation driver=154434 (CANCELLED_BY_USER at step 3235 before checkpoint)
B workstation root=/home/zhiheng/tmp/pf_246cube_6h48h_B_workstation_v1_20260731
B workstation device=NVIDIA GeForce RTX 5080
B workstation initial performance=approximately 1.267 s/step
```

C VIP副本通过同一cluster二进制/参数/fixture预检；提交时第二张A100由
另一用户作业73203占用。B workstation副本使用此前已验收的原生二进制
哈希`7581c169...90602`，其物理源码、参数、fixture、输出时刻和每小时
checkpoint合同不变；启动后GPU 100%、stderr为空。冗余副本不得与原始
B/C输出混写，最终集合统计每个replicate只能选一个通过完整资格的权威
路径，不能把同一replicate的重复运行当成额外随机样本。

2026-07-31用户随后明确取消workstation B，以释放RTX 5080用于弹性求解器
资格测试。driver 154434、main_cuda 154479和GPU监控154477均已TERM退出；
最后写出的日志步为3235，计划的step 3633 checkpoint尚未到达。因此该
workstation B副本只作为被用户取消的非权威部分运行保留，不得用于
6–48 h科学统计。cluster A/B/C及C VIP副本未因此修改。

### 13.14 warm-start＋残差控制弹性求解器资格（2026-07-31）

在不改变物理参数、fixture、dt、热力学、动力学或主研究边界的条件下，
实现并验收：

```text
ELASTIC_WARM_START_RESIDUAL_V1
elastic_warm_start_enabled=1
elastic_residual_control_enabled=1
elastic_iter_min=2
elastic_iter_max=32
elastic_residual_tolerance=1e-6
elastic_residual_absolute_floor=1e-30
elastic_fail_on_nonconvergence=1
```

求解器保留上一步已收敛的GPU k-space位移场，用作下一宏步初值；GPU计算
Hermitian加权相对fixed-point残差，只回传两个标量和。旧固定迭代路径默认
不变。新V4 checkpoint增加完整位移warm state、source-field time level、
迭代/残差和solver fingerprint；V2/V3仍可读，但当新求解器要求弹性状态时
必须fail closed拒绝缺少状态的旧checkpoint。

第一候选保留为失败证据：hard cap 20时首个冷启动步残差
`1.83728973425113764e-06`，高于`1e-6`。未放宽残差要求；最终仅把数值
安全上限提高到32，冷启动在22次以`7.47906089762138573e-07`通过。

最终246³ replicate-B、T380、dt_code=0.02、256步相同fixture对照：

```text
status=PASS_ELASTIC_WARM_START_RESIDUAL_V1
fixed_seconds_per_step=1.2771796875
accelerated_seconds_per_step=0.6447265625
speedup=1.9809633444410786
candidate_mean_elastic_iterations=4.59765625
candidate_mean_warm_iterations=4.529411764705882
candidate_max_iterations=22
nonconverged_steps=0
peak_VRAM_fixed_MiB=4680
peak_VRAM_accelerated_MiB=4854
particle_count=96
candidate_mass_relative_error=1.764705485777036e-13
```

相对固定迭代端点差异：

```text
phi_normalized_L1=6.381899413146483e-07
beta_volume_fraction_relative=1.1573401665436801e-07
mean_radius_relative=4.68456401790241e-08
Sv_relative=8.534742927856135e-08
M6_relative=1.6799146011998583e-07
matrix_xAg_absolute=2.6716003567392455e-09
mean_elastic_energy_relative=2.920337715577279e-07
```

continuous 256与128+restart+128的V4 checkpoint字节一致：

```text
checkpoint_sha256=0cab6cd5f3f8965f99b4b0864afacd84b63f30a863346f343d6faf8a0574151e
checkpoint_bytes=656478472
restart_status=PASS_BYTEWISE
main_cuda_sha256=93e4e6159184de783df406e9c6329459ab0d789026f17adc9bb63757e1f3f385
main_cuda.cu_sha256=e55277be8bb46355f8b0ef4b58e1bebf4f8a5f47331c91e8a3823060cd3f7340
```

最终非覆盖workstation证据根：

```text
/home/zhiheng/tmp/pf_elastic_warm_start_residual_v2_20260731
```

本地冻结证据与详细报告：

```text
reports/pf_elastic_warm_start_residual_v1/
```

该PASS只资格化未来生产可选的数值求解器，不会追溯改变已启动的cluster
A/B/C生产二进制或其冻结参数。GP及所有成核/释放路径继续在主研究中关闭。

## 十四、所有后续工作的强制规则

1. 任何代理开始工作前先读取本文件。
2. 不得把GP、GP Birth、GP release或绝对β首次成核重新加入本阶段主线。
3. 不得用案例专用物理参数调整来消除初态瞬态或匹配48 h。
4. 所有实验比较必须经过APT/PF/TEM观察算子，不得直接一对一比较对象数。
5. 所有生产结果必须记录源码、参数、二进制、初态和分析脚本哈希。
6. 有弹性结果必须报告固定胞/周期边界、eigenstrain取向和弹性能provenance。
7. 颗粒身份遇到merge/split必须fail closed；总体PSD与单粒子轨迹要分开定级。
8. 热输运必须区分PF提供的析出物结构量和外部提供的位错背景。
9. Yu 48 h的0.119等效位错强度仅是诊断，不是可移植物理参数。
10. 大型运行前先做短预检、守恒、dt、restart和输出合同验证。
