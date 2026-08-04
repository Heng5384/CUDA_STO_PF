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

### 13.15 下一任务源码身份固定（2026-07-31）

用户指定下一项新任务必须使用以下Git身份：

```text
branch=codex/pf-dynamic-microstructure-audit-v1
commit=1548895473524add77bd5ee3df967b84f7af8a12
commit_subject=feat(pf): qualify warm-start elastic runtime
```

远端分支头已经核验为该完整提交。下一任务在编译、预检和提交前必须重新
核对`HEAD`及源码树；生产manifest必须记录该提交、源码树、参数、二进制、
fixture和分析器哈希。现有已启动A/B/C任务保持原冻结身份，不得在运行中
替换源码或重新解释其provenance。

该源码固定规则随后用于重新启动workstation B。先前B副本在step 3235
附近无stderr退出，且尚未到step 3633首个checkpoint，因此不能安全续跑；
旧输出根保持不覆盖、不得作为完整生产证据。新B身份为：

```text
source_commit=1548895473524add77bd5ee3df967b84f7af8a12
source_root=/home/zhiheng/tmp/codex_pf_246cube_6h48h_commit1548895_v1_20260731
runtime_binary_sha256=7efa1c075a5476dcac394814a2700d6ba8898da57cb888009d4d0b739c706149
fixture_B_sha256=b37682e5aea7cc294a675ce562a34fb0d306181990d40090df1d20779a93980c
parameter_sha256=ecbdd0ac070bdf5e5d214322b5248a08f5ca5dd4e0670427513f5ef977ea977a
run_root=/home/zhiheng/tmp/pf_246cube_6h48h_B_workstation_commit1548895_v2_20260731
driver_pid=167973
initial_performance=approximately 1.259 s/step
```

新B从原始6 h fixture重新开始，不继承旧未检查点化状态；启动预检PASS、
stderr为空、RTX 5080利用率100%、显存约4680 MiB。

### 13.16 弹性warm-start默认合同（2026-07-31）

用户授权将已验收的`ELASTIC_WARM_START_RESIDUAL_V1`设为所有有弹性运行的
源码和参数转换器默认值：

```text
elastic_warm_start_enabled=1
elastic_residual_control_enabled=1
elastic_iter_min=2
elastic_iter_max=32
elastic_residual_tolerance=1e-6
elastic_residual_absolute_floor=1e-30
elastic_fail_on_nonconvergence=1
```

默认范围包括Ji-Chen零模dynamics、旧非零模dynamics和minimize。三类路径
共享同一GPU fixed-point warm state与Hermitian加权残差停止实现。区别是：

- Ji-Chen零模checkpoint/restart继续保存并严格验证V4完整warm state；
- 非零模dynamics和minimize只使用当前进程内warm state，不宣称跨重启V4；
- `elastic_enabled=0`时该求解器不分配也不运行；
- 显式同时设置两个开关为0可回退历史fixed-iteration路径，回归基线固定
  `elastic_iter_max=20`。

本次没有修改热力学、动力学、弹性常数、eigenstrain、网格、dt或fixture。
本地默认值/参数物化/脚本合同测试、host checkpoint V3/V4测试及workstation
`sm_120` CUDA编译通过。当前已经运行中的A/B/C及workstation B进程不会被
源码默认值追溯修改；只有用新提交重新编译并新启动的任务采用本合同。

用户随后明确取消该workstation B。driver `167973`、`main_cuda 168016`
及其GPU监控进程已定向终止；最后输出约为step 605，未到首个step 3633
checkpoint。输出根完整保留，driver因SIGTERM记录`exit_code=143`。这是
`USER_CANCELLED_BEFORE_FIRST_CHECKPOINT`，不是数值失败，也不得用于生产
统计或替代仍在cluster排队的B。

用户随后指定新Git身份并重新启动workstation B：

```text
source_commit=1c08f9ee011b31e0cd4d82749e58a8f69ebd2204
commit_subject=feat(pf): default to warm-start elasticity
source_root=/home/zhiheng/tmp/codex_pf_246cube_6h48h_commit1c08f9e_v1_20260731
runtime_binary_sha256=ad4a6367650442fd6da2ec36468bf7a04628a62b87a4c662cab1e03b930c2500
fixture_B_sha256=b37682e5aea7cc294a675ce562a34fb0d306181990d40090df1d20779a93980c
parameter_sha256=ecbdd0ac070bdf5e5d214322b5248a08f5ca5dd4e0670427513f5ef977ea977a
run_root=/home/zhiheng/tmp/pf_246cube_6h48h_B_workstation_commit1c08f9e_v3_20260731
driver_pid=169813
elastic_solver_mode=ELASTIC_WARM_START_RESIDUAL_V1
initial_overall_performance=approximately 0.640 s/step
initial_recent_50_step_performance=approximately 0.600 s/step
```

远端分支头、干净source archive、默认合同测试、host V3/V4 checkpoint测试、
`sm_120`编译、fixture/参数/binary预检均PASS。启动后stderr为空、RTX 5080
利用率100%、显存约4854 MiB。该运行从原始6 h B fixture重新开始，并使用
44个约每物理小时checkpoint及冻结的6/12/18/24/36/48 h输出合同。

用户随后要求cluster也切换到同一提交。旧A作业`73230`在step 39963、
11个完整checkpoint后被用户取消，旧输出根完整保留；旧cluster B排队
作业`73231`已取消且没有输出。为避免旧版本C自动抢占GPU，`73232`与
`73235`被可逆地置为`JobHeldUser`，未删除。

新cluster `sm_80`身份与A运行：

```text
source_commit=1c08f9ee011b31e0cd4d82749e58a8f69ebd2204
source_root=/data/home/luozhiheng/tmp/codex_pf_246cube_6h48h_commit1c08f9e_v1_20260731
build_job=73285
build_status=PASS_PF_246CUBE_CLUSTER_BUILD_V1
runtime_binary_sha256=516489b3e4dbafd6ba5876beb2858df8309fbfcbd1065d455b73f1782f5fe8f5
new_A_job=73288
new_A_root=/data/home/luozhiheng/tmp/pf_246cube_6h48h_A_commit1c08f9e_v2_20260731
elastic_solver_mode=ELASTIC_WARM_START_RESIDUAL_V1
initial_cluster_performance=approximately 0.150 s/step
```

默认合同测试、host V3/V4 checkpoint测试、cluster build、fixture/参数/
binary预检均PASS。新A从原始6 h fixture重新开始，stderr为空，A100利用率
约99%、显存约4585 MiB；旧A检查点没有被接入新求解器身份。

用户随后明确要求删除旧A的step 39963结果。精确删除的唯一目标为：

```text
/data/home/luozhiheng/tmp/pf_246cube_6h48h_A_v1_20260731
```

该旧根约5.0 GiB，包含11个旧checkpoint；已永久删除且不可恢复。新A
作业`73288`、新输出根、源码、输入fixture和Slurm日志均未触碰，新A继续
运行。

### 13.17 实验远场锚定A/B/C生产替换（2026-07-31）

用户决定不再把三个246³初态强制投影到
`target_mean_C_Btot=0.03`，而是固定6 h实验matrix浓度：

```text
target_matrix_xAg=0.0062
target_matrix_xB=0.006219279767278563
global_inventory_mode=DERIVED_FROM_EXPERIMENT_MATRIX_AND_FROZEN_RESOLVED_BETA
global_inventory_reprojected_to_0p03=false
```

新fixture保持原A/B/C的96颗粒、中心、半径、冻结库entry、`phi`和局部
`delta_C_relaxation`不变；新全局canonical inventory由实验matrix baseline
和resolved-beta库存自然导出。没有裁剪、归一化或全局质量投影。

```text
A_fixture_sha256=4b3ffa3ab66d36d86c69d7b24505e04acf19a64311ad19a2ed10e48be0c16937
A_mean_C_Btot=0.029422681887673947
B_fixture_sha256=05200bd91b3df72a9a1adab347320a5523020f8b768bd1ca38315572b5ed3c7d
B_mean_C_Btot=0.029422741495106243
C_fixture_sha256=f053e80717a0b8c45794e38df40818905964f83ca14534484355a2146a384cb7
C_mean_C_Btot=0.029422676887399385
```

三者全字段哈希、96颗粒身份、禁止路径和machine-precision质量分解审计
PASS。`h<1e-4`初始matrix观测`xAg`为0.00619852–0.00619865；相对0.0062
的小差异来自保留的局部弹性松弛尾场。共同多颗粒平衡仍不作声明，初始
重构继续计入6 h后的真实物理演化。

生产身份和登记：

```text
base_source_commit=1c08f9ee011b31e0cd4d82749e58a8f69ebd2204
parameter_sha256=ecbdd0ac070bdf5e5d214322b5248a08f5ca5dd4e0670427513f5ef977ea977a
cluster_binary_sha256=516489b3e4dbafd6ba5876beb2858df8309fbfcbd1065d455b73f1782f5fe8f5
workstation_binary_sha256=ad4a6367650442fd6da2ec36468bf7a04628a62b87a4c662cab1e03b930c2500
A_gpu_uvip_job=73301
B_gpu_vip_24h_job=73302
B_gpu_uvip_fallback_job=73304
C_workstation_driver=175666
C_gpu_uvip_fallback_job=73309
```

B和C的重复候选均使用不同非覆盖根；每个replicate最终只能选择一个完整
PASS路径作为权威结果，不得把重复运行计为额外随机样本。新任务预检和提交成功后，旧
cluster作业`73288/73232/73235`已取消；旧workstation B在完整step 10899
checkpoint后取消，step 14532段未完成。所有现有旧输出均保留，未删除。

完整启动登记：

```text
reports/pf_246cube_experiment_matrix_handoff_v1/launch_20260731.md
```

### 13.18 PF full-PSD无位错晶格输运接口V1（2026-07-31）

已停止继续拟合Yu位错参数。公开参数复现和
`s_dis=0.1172768`仅作为隔离的历史诊断证据保留，未覆盖旧报告，也未进入
当前PF输运计算。新接口正式冻结：

```text
A_N=1.5
dislocation_mode=DISLOCATION_OFF
SI_S11_rate=0
SI_S13_rate=0
yu_refit_scale_used=false
```

Yu S7--S10的单平均半径析出物公式已扩展为resolved PF full-PSD直接求和：

```text
tau_Pre^-1(omega,t)=v/V_box*sum_i[(sigma_S(R_i)^-1+sigma_l(R_i,omega)^-1)^-1]
```

每个transport snapshot保存完整粒径列表、`Nv`、球形等效`Sv`、`M6`、
matrix `xAg/xB`、时间、盒体积以及源码、参数、二进制、fixture和分析器
provenance。非PF host参数冻结为Yu公开48 h非颗粒baseline；Yu小/大单半径
颗粒人口被PF full PSD替换。matrix `xAg`只更新点缺陷`Gamma`，没有引入
成分依赖晶格、晶粒、声速或Debye温度拟合。

两条科学路径定义为：

```text
delta_kappa_PSD = kappa_no_dis[PSD(t),xAg(6h)]-kappa_no_dis[PSD(6h),xAg(6h)]
delta_kappa_PSD_plus_matrix = kappa_no_dis[PSD(t),xAg(t)]-kappa_no_dis[PSD(6h),xAg(6h)]
```

历史有弹性6--48 h、43快照轨迹已完成接口smoke。该轨迹只资格化接口，
不得替代实验matrix锚定A/B/C ensemble。其总体PSD观察量可用；旧严格
merge阻塞和后续merge-aware增量证据同时保留，不宣称merge后的独立颗粒
身份轨迹。

全部资格门通过：

- 单分散退化到Yu单半径公式；
- full PSD向量化与逐颗粒直接求和；
- PSD bin细化；
- `Sv+M6`等效单分散矩重建；
- 256/512点与独立自适应Debye积分；
- `Nv/Sv/M6/mean-R`源数据闭合；
- 两次生成逐文件SHA-256确定性；
- A/B/C零选择、多选择、重复或非完整权威轨迹fail closed。

关键数值上限：

```text
direct_sum_max_relative_difference=3.254921596850476e-16
source_descriptor_closure_max_relative=8.490224733760293e-16
gauss_256_512_max_relative_difference=7.32232010751243e-15
gauss_adaptive_max_relative_difference=2.7631396632122253e-15
synthetic_psd_bin_0p125nm_kappa_relative_error=1.6272503690062254e-06
```

最终接口状态：

```text
PASS_PF_FULL_PSD_NO_DISLOCATION_TRANSPORT_INTERFACE_V1
```

生产ensemble仍为：

```text
PENDING_EXPERIMENT_MATRIX_ANCHORED_A_B_C_COMPLETE_PASS_AUTHORITIES
```

A/B/C完成后，每个replicate必须且只能选择一条完整PASS权威轨迹，再输出
`kappa_L_no_dis(T,t)`、`delta_kappa_PSD`、
`delta_kappa_PSD_plus_matrix`的均值、样本标准差和范围。当前不得把历史
smoke数值解释为绝对实验热导率复现。

冻结实现与证据：

```text
scripts/pf_full_psd_no_dislocation_transport_v1.py
scripts/qualify_pf_full_psd_no_dislocation_transport_v1.py
scripts/assemble_pf_full_psd_no_dislocation_transport_ensemble_v1.py
data/qualification/pf_full_psd_no_dislocation_transport_v1/transport_parameter_contract.json
reports/pf_full_psd_no_dislocation_transport_v1/

transport_interface_script_sha256=a42f933c43d0adbd4fc9a4d86a7b74bc08d4efe64517cc3fbf1ba177b62dcf66
qualification_script_sha256=d37170711e2d03b8fb9c178548e8e2f35e49a2619947281ace76bb27300da884
ensemble_assembler_script_sha256=a2a861722d065123f62fba46e1da663c56f34b67fa9832d9797670b53091b9a2
transport_parameter_contract_sha256=d16e948dd34403129b2deecb90efbea937251e23ae9c128c353b6f66bbee1ae0
```

### 13.19 四分之一纳米profile库、正式方法1与A/B/C生产替换（2026-08-01）

在相同源码树和相同弹性最小化合同下，原有0.5 nm库已补齐为
8.00--11.50 nm、0.25 nm间隔的15-entry单颗粒target-profile库。7个新增
中间半径全部通过质量、收敛、stderr和哈希审计；组合库状态为：

```text
library_status=PASS_PF_ELASTIC_TARGET_PROFILE_LIBRARY_V1
library_manifest_sha256=de4142e0268e379f70fd1c860aab9e004421d07df4f8d872f4eaeb3dbef7af5b
source_tree_sha256=f7855699addf98f9d5aed03af876d851d524fb62c98f5a48df78d1fe561a2a75
```

正式方法1使用整数PSD优化，仅重新选择库entry及其颗粒数量，不缩放、插值
或改写任何单颗粒profile。96颗粒选择得到：

```text
target_h_volume_nm3=356237.61016735336
selected_h_volume_nm3=356237.6264614671
absolute_h_volume_error_nm3=0.01629411377
selection_sha256=57e948762f97011116eeba61b87dcf19c16fffb755357ac228742419585e3076
optimizer=EXACT_MONOTONE_INTEGER_DYNAMIC_PROGRAM_V1
```

据此组装的A/B/C 246^3静态fixture均通过逆序确定性、96颗粒、全局
`mean_C_Btot=0.03`、实验matrix浓度及machine-precision库存审计：

```text
A_fixture_sha256=f6cce1bd1abaf0e8767f52fc70009a1cce9c928a442bc3ebd38e7c8e7c26d7b2
A_matrix_xAg=0.006202336869744194
B_fixture_sha256=8106091d92b2924f65a23ab5adf8e98d958dd6bac63f3fb099c4f85567c40389
B_matrix_xAg=0.006202284267465961
C_fixture_sha256=2ad665a3b58d26c29fcfdbfc2c949be979e602147283fce25bc00d992acaac64
C_matrix_xAg=0.006202343009149754
static_status=PASS_246CUBE_QUARTER_NM_HANDOFF_STATIC_V1
```

必要字段已由workstation经内网传至cluster并逐文件校验。新A/B/C均提交到
`gpu_uvip`，使用相同`sm_80`二进制、参数、15-entry库和非覆盖输出根：

```text
runtime_binary_sha256=516489b3e4dbafd6ba5876beb2858df8309fbfcbd1065d455b73f1782f5fe8f5
parameter_sha256=ecbdd0ac070bdf5e5d214322b5248a08f5ca5dd4e0670427513f5ef977ea977a
dt_code=0.02
dt_physical_s=0.9909260953431841
final_step=152585
checkpoint_cadence=3633
A_job=73364
B_job=73365
C_job=73366
```

提交新任务后，旧任务`73301/73302/73304/73309`按用户要求取消；旧输出
未删除。新A已开始运行，初始稳定吞吐约0.15--0.16 s/step，stderr为空，
`mean_C_Btot`保持0.03；B/C因`gpu_uvip`单用户并发限制串行排队。动态
6--48 h结果仍为RUNNING/PENDING，不得把静态组装PASS解释为生产科学PASS。

启动登记：

```text
reports/pf_246cube_quarter_nm_method1_v1/assembly_and_launch_20260801.md
```

### 13.20 246³生产PASS到full-PSD无位错输运的严格入口（2026-08-01）

历史transport smoke与246³正式生产审计使用不同字段和不同权威合同。现已
新增独立适配器，只允许一条完整生产PASS转换成transport snapshot，不修改
历史smoke、Yu公开参数复现或`s_dis=0.1172768`诊断证据。

入口强制要求：

- `status.txt`和`audit.json`均为精确
  `PASS_246CUBE_6H48H_CONDITIONAL_PRODUCTION_V1`；
- production audit全部gate为真，且merge-aware粒子谱门为PASS；
- trajectory class明确包含`EXPERIMENT_MATRIX_ANCHORED`；
- A/B/C标签、fixture哈希、campaign manifest互相一致；
- 只接受step `0/21798/43596/65393/108989/152585`，注册为
  `6/12/18/24/36/48 h`；实际PF elapsed time和source experimental age另存
  于provenance；
- `registered_observables.csv`与`registered_particle_psd.csv`逐时刻颗粒数
  闭合，并由逐颗粒半径重算`Nv/Sv/M6/mean-R`；
- source commit、运行binary、参数、fixture、analysis binary、输入哈希账本、
  production audit和campaign manifest全部保留路径与SHA-256；
- 禁止GP、GP Birth、GP release、external source和new-beta nucleation。

生产格式适配器的独立资格测试已PASS。正向测试使用历史smoke派生的六时刻
生产schema合成fixture，不使用任何A/B/C生产科学数据；两次转换的全部六个
transport输出逐文件哈希一致。非实验matrix轨迹、merge-aware失败、PSD缺行
和fixture哈希不一致均fail closed。

```text
status=PASS_PF_246CUBE_NO_DISLOCATION_TRANSPORT_AUTHORITY_ADAPTER_V1
max_descriptor_closure_relative=8.790736184043864e-16
production_A_B_C_data_used=false
running_PF_jobs_modified=false

production_adapter_script_sha256=acd5fb5a112b68b00dcf85e37da39ea124820b443edf3d9314d1b4c129ad2cec
production_adapter_qualification_script_sha256=f0c224133586d011684a0ba8374fb54122d12d1f3c1b7965c0635aeda19121ce
production_adapter_qualification_json_sha256=cd17a90a88a0c5d162273ce9cec20ddbb62570d7b8af99635fbbd6b1fe519773
```

正式ensemble仍不得生成。当前权威候选是13.19登记的四分之一纳米Method-1
`73364/73365/73366`；在每个replicate形成唯一完整production PASS并经本入口
转换前，状态保持：

```text
PENDING_EXPERIMENT_MATRIX_ANCHORED_A_B_C_COMPLETE_PASS_AUTHORITIES
```

### 13.21 输运生产权威链与ensemble网格加固（2026-08-01）

13.20的生产适配入口和13.18的A/B/C ensemble选择器已进一步fail closed
加固，原历史smoke、旧资格目录和Yu诊断证据均未覆盖。

单replicate生产入口现在除精确production PASS、全真gate、merge-aware PASS、
六个注册科学时刻和full-PSD矩闭合外，还强制保存完整44-checkpoint SHA-256
链；缺少任一checkpoint哈希即拒绝。ensemble选择器不再只验证transport
输出格式，而是同时要求：

- 40字符完整Git source commit；
- 精确`PASS_246CUBE_6H48H_CONDITIONAL_PRODUCTION_V1` provenance；
- production全部gate为真及merge-aware gate为PASS；
- source binary、parameter、analysis binary、fixture、raw observables、raw PSD、
  production audit、campaign、input hash ledger和production adapter的合法
  SHA-256；
- production adapter哈希必须与当前冻结脚本逐字节一致；
- 每个replicate恰有6个注册snapshot；
- `kappa`逐个A/B/C×age×temperature单元恰好一行；
- descriptor逐个A/B/C×age×temperature×matrix-mode×descriptor单元恰好一行，
  缺行和重复行均拒绝。

新的非覆盖资格目录全部PASS。生产入口测试增加“44-checkpoint链缺一个哈希”
负例；完整接口资格增加“缺少production PASS provenance”和“descriptor少一个
网格单元”负例。两项均按预期fail closed；合成正向A/B/C仍得到
`PASS_PF_FULL_PSD_NO_DISLOCATION_TRANSPORT_ENSEMBLE_V1`。本资格不使用真实
A/B/C生产科学结果，不改变任何运行中PF。

```text
interface_status=PASS_PF_FULL_PSD_NO_DISLOCATION_TRANSPORT_INTERFACE_V1
production_ingress_status=PASS_PF_246CUBE_NO_DISLOCATION_TRANSPORT_AUTHORITY_ADAPTER_V1
production_ensemble_status=PENDING_EXPERIMENT_MATRIX_ANCHORED_A_B_C_COMPLETE_PASS_AUTHORITIES

production_adapter_script_sha256=6e85ecb29f767fe96fe2ff4e9262ddf547629a6d538cc880d4d6d829a021104d
ensemble_assembler_script_sha256=84763345be6726b419e0cccf093880f328f1854f91ba5a6ecf3c127692728ce0
production_adapter_qualification_script_sha256=17a160d94c076379d5563b9e9b587c9182b845c1e7c679a657b7100b28c8f7f6
full_interface_qualification_script_sha256=8b289bf18c15529976b81e69d285ef76db5b443ac7d0f46364259f6ac0f2f183
production_adapter_qualification_json_sha256=fc3356b0b3c3d3e96cf9d8f4119a97a1e520ee413e09b60052ab7b3997ad8887
full_interface_qualification_json_sha256=abdd85f9a6e59369674533b980c1d4215901a4c2c22dbea76453b8d5b7a94d32
pending_method1_authority_manifest_sha256=05a5182274ff2258713436bf628014363b0ca7b9f943e88f2da8e677a580c984
```

当前A/B/C Method-1候选身份已写入非权威pending manifest；该文件故意保持
零选择，并已验证会被ensemble选择器拒绝。只有真实三条production PASS经
适配器生成transport manifest后，才能复制为最终authority manifest并逐条
显式选择。

### 13.22 旧A/B/C真实6--8 h描述量充分性比较（2026-08-01）

已对13.12中同一批真实246³ A/B/C短筛选数据完成只读输运描述量比较。
输入共51个快照（每个replicate 17个），三条6--8 h短筛选均PASS；C的唯一
diffuse-tail merge使用已冻结的merge-aware PASS。分析只使用每个快照的总体
resolved-particle PSD，不新增merge后独立粒子身份声明，也未修改正在运行的
Method-1 A/B/C任务。

五种比较为：

```text
Nv + mean_R = measured Nv and mean R, monodisperse closure
Nv + mean_R + CV = measured Nv/mean/CV, lognormal closure
Sv = established geometric scattering limit
Sv + M6 = equivalent population matching both moments
full_PSD = direct resolved-particle sum and reference
```

冻结Yu 48 h非颗粒host、`A_N=1.5`、位错S11/S13关闭、无refit scale；计算
300--600 K七个温度。`fixed_6h_matrix`隔离PSD贡献，
`pf_time_varying_matrix`同时把PF远场Ag传入点缺陷散射。

聚合误差（full PSD为参考）：

| 描述量 | fixed matrix κ MAPE | fixed Δκ NRMSE | varying matrix κ MAPE | varying Δκ NRMSE |
|---|---:|---:|---:|---:|
| \(N_v+\bar R\) | 0.2872% | 349.7% | 0.2890% | 30.17% |
| \(N_v+\bar R+CV\) lognormal | 0.03274% | 57.42% | 0.03305% | 4.977% |
| \(S_v\) geometric | 7.134% | 889.8% | 7.195% | 77.38% |
| \(S_v+M_6\) | 0.08846% | 82.54% | 0.08901% | 7.114% |
| full PSD | 0 | 0 | 0 | 0 |

因此在本2 h窗口内，加入CV使`Nv+mean_R`的绝对κ平均误差降低约8.7倍，
也是幅值误差最小的压缩表示；`Sv`单独使用不足；`Sv+M6`次优，并在
time-varying matrix比较中保持了全部A/B/C Δκ排序，而lognormal闭合的精确
排序比例为0.625。绝对κ误差会被共同host散射稀释；fixed-matrix Δκ信号很
小，因此压缩模型的演化信号相对误差明显放大。

当前不得冻结`Nv+mean_R+CV`为最终最小充分描述量，原因是：

- 三个矩不唯一决定PSD，当前结果依赖显式lognormal形状闭合；
- fixed-matrix微观结构信号NRMSE仍约57%；
- 6--8 h初始PSD较窄，不能代表6--48 h后期宽PSD；
- 该旧fixture的6 h远场Ag约0.006788，不是当前Method-1实验锚定值0.0062。

正式合同仍为：full PSD作为生产权威输入；等13.19 Method-1 A/B/C完整
6--48 h PASS后，在同一分析器下重复五模型比较，再决定最小充分描述量。

```text
status=PASS_PF_ABC_6H8H_DESCRIPTOR_SUFFICIENCY_V1
snapshot_count=51
prediction_cell_count=3570
source_descriptor_closure_max_relative=1.25931189395712e-15
lognormal_mean_closure_max_relative=2.706509665625206e-15
lognormal_CV_closure_max_absolute=7.494005416219807e-16
deterministic_regeneration_status=PASS_IDENTICAL_FILE_SHA256
full_PSD_reference=true
running_PF_jobs_modified=false
absolute_experimental_kappa_claim=false
```

冻结实现与证据：

```text
scripts/analyze_pf_abc_6h8h_descriptor_sufficiency_v1.py
reports/pf_abc_6h8h_descriptor_sufficiency_v1/
```

### 13.23 Yu 2024原始Debye--Callaway计算合同与复现边界（2026-08-01）

本节冻结Yu等人在Advanced Energy Materials 2024论文中的原始晶格热输运
计算流程，并与13.18的PF full-PSD接口严格区分。Yu原模型分别计算AQ和
48 h两个实验状态，不是从6 h动态积分到48 h的微观结构演化模型。

原始资料：

```text
doi=10.1002/aenm.202304442
main_pdf_sha256=da838898f561a26b00b37124b18f18df627adf95bf9fe4622f3089f3aaa19eba
supporting_information_pdf_sha256=f60bef889c0445b790f9a93828aa40b7ca56bf40850a461b0aff019a05d37ff0
```

#### 13.23.1 从实验总热导率到晶格热导率

Yu先由实验总热导率扣除电子热导率：

```text
kappa_ele = L*sigma*T
kappa_lat,experimental = kappa_tot-kappa_ele
```

其中Lorenz数`L`由single-parabolic-band模型计算。随后以实验缺陷、晶粒、
析出物和位错参数构造Debye--Callaway模型，与实验`kappa_lat(T)`比较。

#### 13.23.2 Debye积分和总散射率

公开模型使用单总松弛时间Debye积分，不包含标准Callaway模型单独的
Normal-process第二积分：

\[
\kappa_{\rm lat}(T)=
\frac{k_B}{2\pi^2v}
\left(\frac{k_BT}{\hbar}\right)^3
\int_0^{\Theta_D/T}
\tau_{\rm tot}(x,T)
\frac{x^4e^x}{(e^x-1)^2}\,dx,
\qquad
x=\frac{\hbar\omega}{k_BT}.
\]

散射率按Matthiessen规则相加：

\[
\tau_{\rm tot}^{-1}=
\tau_{U+N}^{-1}+
\tau_{GB}^{-1}+
\tau_{PD}^{-1}+
\tau_{Pre}^{-1}+
\tau_{Dis}^{-1}.
\]

每个温度的实际计算顺序为：在`0 <= x <= Theta_D/T`建立求积节点，将
`x`转换为`omega=x*k_B*T/hbar`，逐节点计算全部散射率，相加后取倒数得到
`tau_tot`，最后执行Debye积分。

#### 13.23.3 各散射机制

Normal和Umklapp过程由SI S1合并，公开信息不能将二者单独分解：

\[
\tau_U^{-1}+\tau_N^{-1}=
A_N\frac{2}{(6\pi^2)^{1/3}}
\frac{k_B\bar V^{1/3}\gamma^2\omega^2T}{\bar Mv^3}.
\]

`A_N=1.5`是Table S2唯一明确标为`fitted`的因子。平均原子体积使用
`Vbar=a_i^3/8`。

晶界散射为：

\[
\tau_{GB}^{-1}=v/d.
\]

点缺陷散射为：

\[
\tau_{PD}^{-1}=\frac{\bar V\omega^4}{4\pi v^3}\Gamma,
\qquad
\Gamma_i=x_i\left[
\left(\frac{\Delta M_i}{M}\right)^2+
\epsilon\left(\frac{\Delta\delta}{\delta}\right)^2
\right],
\qquad
\Gamma=\sum_i\Gamma_i.
\]

公开SI只计入间隙Ag并冻结`epsilon=65`。Table S2把`Delta_M_i`描述为
杂质与基体的质量差，但列值为`107.87 g/mol`，等于Ag原子量；正式字面
复现保留该公开数值，不擅自改成Pb--Ag差值。

对一个平均半径`R`、数密度`N_P`的析出物人口：

\[
\tau_{Pre}^{-1}=vN_P\sigma_{eff},
\qquad
\sigma_{eff}=(\sigma_S^{-1}+\sigma_l^{-1})^{-1},
\]

\[
\sigma_S=2\pi R^2,
\qquad
\sigma_l=\frac49\pi R^2
\left(\frac{\Delta D}{D_M}\right)^2
\left(\frac{\omega R}{v}\right)^4,
\qquad
\Delta D=D_{PbTe}-D_{Ag_2Te}.
\]

该调和截面连接低频Rayleigh极限和高频几何极限。Yu分别计算small和big
两个单平均半径人口，再把两者散射率相加；原模型没有使用连续完整PSD。

位错芯与普通位错应变场为：

\[
\tau_{DC}^{-1}=N_D\frac{\bar V^{4/3}}{v^2}\omega^3,
\]

\[
\tau_{DS}^{-1}=CB_D^2N_D\gamma^2\omega F,
\]

\[
F=\left[
\frac12+
\frac1{24}\left(\frac{1-2r}{1-r}\right)^2
\left(1+\sqrt2\left(\frac{v_L}{v_T}\right)^2\right)^2
\right].
\]

对48 h Ag-decorated dislocation，SI S13用`gamma+gamma_prime`替代
`gamma`：

\[
\tau_{DS}^{-1}=CB_D^2N_D(\gamma+\gamma')^2\omega F,
\]

\[
\gamma'=\frac{V_mc_iB}{k_BT_a}(\gamma\alpha^2-\alpha\beta),
\qquad
\alpha=\frac{V_i-V_m}{V_m},
\qquad
\beta=\frac12\frac{M_m-M_i}{M_m}.
\]

#### 13.23.4 公开参数身份

主要Table S2输入冻结如下：

| 参数 | AQ | 48 h或共同值 |
|---|---:|---:|
| `A_N` | 1.5 fitted | 1.5 fitted |
| `a_i` | 6.4286 Å | 6.445 Å |
| `gamma` | 1.96 | 1.96 |
| `Mbar` | `2.784e-25 kg` | 同左 |
| `v` | `1770 m/s` | 同左 |
| `Theta_D` | 136 K | 同左 |
| grain size `d` | 10.4 µm | 12.1 µm |
| solid-solution Ag `x_i` | 0.0069 | 0.0034 |
| `R_small` | 2 nm | 2 nm |
| `N_small` | `7.5e24 m^-3` | `2.5e19 m^-3` |
| `R_big` | 30 nm | 50 nm |
| `N_big` | `1.9e21 m^-3` | `9.2e18 m^-3` |
| `N_D` | 0 | `3.5e11 cm^-2` |

共同位错参数包括`B_D=4.56e-10 m`、`C=0.96`、`r=0.218`、
`v_L=3590 m/s`、`v_T=1610 m/s`。Ag偏聚项采用`c_i=4 at.%`、
`B=4.1e10 Pa`、`T_a=655 K`和`gamma_prime=2.62`。SI将`V_m/V_i`
单位印为`m^-3`，维度上应为每原子`m^3`；380 °C为653 K而SI列655 K，
均作为公开来源歧义记录，不静默改写。

Yu的谱贡献图通过逐项加入散射机制形成：`U+N`，再加`GB`、`PD`、
small precipitate、big precipitate以及dislocation；相邻曲线差用于解释各
机制的相对贡献。AQ的低热导主要归因于极高数密度小Ag-rich对象和大析出物，
点缺陷主要影响高频声子；48 h粗化后析出物人口显著下降，公开解释转为
Ag-decorated dislocation strain主导。

#### 13.23.5 独立复现结果和不可越过边界

严格使用公开公式和Table S2字面参数的复现结果为：

```text
AQ_MAPE=0.545_percent
AQ_max_relative_error=1.011_percent
AQ_status=PASS_PUBLIC_MODEL_REPRODUCTION

48h_strict_S13_MAPE=43.348_percent
48h_strict_S13_max_relative_error=48.640_percent
48h_status=BLOCKED_PUBLIC_MODEL_CURVE_MISMATCH
```

约303.94 K时，严格S13计算为`1.0268 W m^-1 K^-1`，论文数字化模型曲线
为`1.9993 +/- 0.0113 W m^-1 K^-1`，实验约`2.04 W m^-1 K^-1`；关闭位错
时为`2.4533 W m^-1 K^-1`。普通S12位错项仅作为诊断时MAPE约8.04%，仍未
通过5%门且与SI要求使用S13冲突。约`0.119`等效位错散射强度只能说明公开
曲线对应的反演尺度，不能宣称作者拟合了位错密度，也不能移植为本项目参数。

复现证据：

```text
reports/yu2024_debye_callaway_reproduction_v1/equation_contract.md
reports/yu2024_debye_callaway_reproduction_v1/parameter_provenance.md
reports/yu2024_debye_callaway_reproduction_v1/reproduction_report.md

equation_contract_sha256=43500490574e2b0a557d636a22fced4e0be81fe85b33e12043a1dda5d931fe8a
parameter_provenance_sha256=3fadc21c3b3d59596c8d8a4d56db7b314398750c41422e4ba9abbbd8fa437c2e
reproduction_report_sha256=7080e8ea6714bc1e15db7726bc9883418062ad4905bf6556aab16b7c1a67cbc0
```

#### 13.23.6 与当前PF项目的正式接口

Yu中的`N_P`是单位物理体积的实验散射对象数，`R`是该人口的平均半径；
它不自动等于PF resolved stoichiometric beta连通域。APT Ag-rich objects、
TEM可见析出物与PF resolved beta必须继续通过第八节观察算子区分。

当前项目只继承Yu公开的非颗粒host散射框架和S7--S10截面物理：

```text
A_N=1.5
DISLOCATION_OFF
Yu_small_big_population_replaced_by_PF_full_PSD=true
case_specific_refit=false
absolute_experimental_kappa_claim=false
```

析出物项使用13.18冻结的逐颗粒直接求和：

\[
\tau_{Pre}^{-1}(\omega,t)=
\frac{v}{V_{box}}
\sum_i
\left[\sigma_S(R_i)^{-1}+\sigma_l(R_i,\omega)^{-1}\right]^{-1}.
\]

因此当前输出是“Yu散射物理框架 + PF完整PSD + 无位错条件”的条件热输运
趋势，不是Yu 48 h绝对实验热导率复现。完整PSD继续作为生产权威输入；
`Nv+mean_R`、`Nv+mean_R+CV`、`Sv`和`Sv+M6`仅用于描述量充分性测试。

### 13.24 A场每小时merge-aware与溶解资格审计（2026-08-01）

在不重跑A场PF的前提下，已对13.19 Method-1 A场的初态及44个每小时
检查点完成三阈值、周期、merge-aware身份和溶解区间审计：

```text
low_support_h_threshold=1e-4
resolved_core_h_threshold=1e-3
strong_core_h_threshold=5e-3
event_time_semantics=BOUNDED_BY_CONSECUTIVE_SNAPSHOTS
exact_event_time_claimed=false
```

旧tracker要求消失前至少三次连续体积递减；小时采样下，很多小颗粒在两个
快照间消失，因此旧BLOCKED不能直接解释为身份或质量失败。新合同只在
`h>1e-3`核心消失同时具有不晚于它的`h>5e-3`强核心消失证据时登记溶解
区间；`h>1e-4`只作为弥散support诊断，不单独定义resolved core。

merge后阈值颈部重新打开时，只有周期质心到最后独立身份锚点形成唯一
一一映射且不跨越Voronoi门，才允许恢复原身份；不按体积排序。极弱交叉
重叠边必须同时不满足主导重叠门并与同一步伪split成对，才可登记为
`INCIDENTAL_WEAK_OVERLAP_EDGE_PRUNED`。任何无法唯一恢复的split、新component、
身份重现、账本缺失、阈值顺序倒置或质量越界继续fail closed。

A场结果：

```text
status=PASS_246CUBE_HOURLY_MERGE_DISSOLUTION_AUDIT_V1
snapshot_count=45
initial_identity_count=96
final_resolved_core_identity_count=5
resolved_core_dissolution_count=91
threshold_contact_group_count=1
persistent_strong_core_merge_group_count=0
unresolved_split_count=0
new_component_count=0
max_mass_relative_error=3.272655446666275e-13
```

91个身份中，30个同时通过旧连续递减证据，60个由三阈值同小时区间完全
消失闭合，1个`P078`为strong/medium核心消失但低幅尾部仍附着于`P087`。
`P078/P087`在低/中/强阈值约16/19/22 h依次接触，强/中阈值约24/25 h
重新分开，随后`P078`核心在25--26 h区间消失。因此该拓扑严格登记为：

```text
THRESHOLD_CONTACT_NOT_PHYSICAL_MERGE
```

不能把它计为真实颗粒合并。48 h低阈值为5个support连通域、承载6个历史
身份；中/强阈值为5个resolved core身份。

验证包括合成正例、9类fail-closed负例、历史真实C 6--8 h回归、本地双运行
逐文件确定性以及cluster CPU作业73425。73425退出码0、stderr为空，全部输出
SHA-256与本地逐字节一致。未使用GPU、未改变A/B/C生产任务或原始输出。

```text
audit_script_sha256=e86bfea8a9088f4980b33b98f32239783a81adbc9b2cc81c946ec51d9e2317f9
test_script_sha256=2f50d4b4f03f04b6ad1e3f7457d1b5106a601bda5fe045f9fa4391f12e2fa045
audit_json_sha256=9f487839ec75c626577e72b180ed9435bb75080022ea8cc7c5e928fa29e41d98
reports=reports/pf_246cube_hourly_merge_dissolution_v1/
```

本PASS只验收A场谱系/contact分类和溶解区间，不覆盖A原始运行根中的旧
BLOCKED文件，也不自动等于完整A生产PASS。生产总判定必须显式接纳本合同，
再与质量、零模、checkpoint、能量和科学观察量共同组装。

### 13.25 B场每小时谱系审计与A/B条件路径比较（2026-08-01）

B场没有重跑PF；使用与13.24完全相同的审计器和三阈值合同，只读初态及
44个每小时checkpoint。Cluster CPU作业73426运行3 min 36 s、退出码0、
stderr为空；新生成的低阈值五个tracker文件与B原生产后处理逐文件SHA-256
一致，本地双审计也逐文件确定。

```text
B_status=PASS_246CUBE_HOURLY_MERGE_DISSOLUTION_AUDIT_V1
B_initial_identity_count=96
B_qualified_dissolution_count=87
B_final_low_support_component_count=8
B_final_strong_identity_lower_bound=8
B_final_medium_identity_upper_bound=9
B_threshold_contact_group_count=2
B_persistent_strong_core_merge_group_count=0
B_unresolved_split_count=0
B_new_component_count=0
B_max_mass_relative_error=2.8581972897407958e-13
```

B的两个组均为阈值接触，不是持续强核心merge：`P077/P086`只在低阈值
约26--32 h连接；`P078/P079`在低/中/强阈值约14/16/19 h依次连接，强阈值
约20 h重新分开。48 h时后者的中/低support仍包含两个历史身份，但强阈值
只保留`P079`；所以B必须报告8个确定强核心、最多9个中阈值历史身份，不能
把9写成精确独立粒子数，也不能把`P078`登记为已完全溶解。

A/B均通过相同谱系合同，初始96颗粒、平均半径、β体积分数和库存几乎相同。
48 h对比为：

| 观察量 | A | B |
|---|---:|---:|
| 低阈值连通域 | 5 | 8 |
| 平均半径 (nm) | 24.8599 | 20.1015 |
| β体积分数 | 0.02383034 | 0.02366171 |
| `Sv` (nm^-1) | 0.00270224 | 0.00301879 |
| `M6` (nm^3) | 116.5319 | 87.0674 |
| 远场Ag (at.%) | 0.630082 | 0.647421 |

A是粗化推进更充分的路径；B保留更多、更小的颗粒以及更高界面面积和基体
Ag。两条48 h远场Ag均在实验`0.62±0.04 at.%`范围内。12 h时B颗粒更少、
平均半径更大，而18 h后B颗粒数反超A，因此不能把差异简化为B全程更慢或
更快。A/B只证明hash-pinned空间实现会显著影响后期PSD、`Sv`和`M6`；两个
样本不能给出ensemble均值或统计收敛，必须等待C完成相同审计。

```text
comparison_status=PASS_246CUBE_HOURLY_MERGE_DISSOLUTION_AB_COMPARISON_V1
B_audit_json_sha256=10185625c90f18d2336515b90b3ac8e21ea1efbc3b464b5a9cacb4342e83c95a
reports_B=reports/pf_246cube_hourly_merge_dissolution_B_v1/
reports_AB=reports/pf_246cube_hourly_merge_dissolution_AB_v1/
```

### 13.26 C场谱系审计与A/B/C初步ensemble（2026-08-01）

C场PF已完整生成6 h初态和44个每小时checkpoint。原作业73366最后的
`BLOCKED_246CUBE_6H48H_PRODUCTION_DRIVER_V1`只来自旧低阈值tracker无法
解释持续merge；不是PF场、checkpoint或质量失败。没有重跑C场；使用与A/B
完全相同的三阈值审计器只读检查已有输出。Cluster CPU作业73448运行
3 min 37 s、退出码0、stderr为空；重建的低阈值五个tracker文件与原C生产
输出逐文件SHA-256一致，本地双审计与cluster结果也逐字节一致。

```text
C_status=PASS_246CUBE_HOURLY_MERGE_DISSOLUTION_AUDIT_V1
C_initial_identity_count=96
C_qualified_dissolution_count=91
C_final_low_support_component_count=3
C_final_surviving_historical_identity_count=5
C_threshold_contact_group_count=0
C_persistent_strong_core_merge_group_count=2
C_unresolved_split_count=0
C_new_component_count=0
C_max_mass_relative_error=3.6428004673166977e-13
```

C的`P075/P081`约在7--9 h由低到强阈值依次连通，`P090/P092`约在
24--27 h依次连通；两组随后都不再分开且重叠证据充分，严格登记为资格通过
的持续强核心merge。故48 h必须同时报告3个物理连通析出体和5个历史幸存
身份；不能把5写成5个独立粒子，也不能把两次merge误计为溶解。

C从6到48 h的主要变化为：连通体96到3、平均半径9.54161到29.8101 nm、
`Sv`从0.00742617降到0.00230167 nm^-1、`M6`从5.35782增到194.638 nm^3；
β体积分数最终只降低0.0579%，远场Ag为0.621465 at.%并位于实验带。C支持
近恒定库存下的强粗化，但因存在两次真实merge，不能单独称为纯LSW路径。

A/B/C均通过相同谱系合同，初始颗粒数、PSD统计、β体积分数和远场Ag实际
等价。48 h初步三样本统计为：

| 观察量 | 均值 | 样本标准差 | 最小--最大 |
|---|---:|---:|---:|
| 连通析出体数 | 5.333 | 2.517 | 3--8 |
| `Nv` (m^-3) | 3.583e20 | 1.690e20 | 2.015e20--5.374e20 |
| 平均半径 (nm) | 24.924 | 4.855 | 20.102--29.810 |
| β体积分数 | 0.0238026 | 0.0001293 | 0.0236617--0.0239158 |
| `Sv` (nm^-1) | 0.00267424 | 0.00035938 | 0.00230167--0.00301879 |
| `M6` (nm^3) | 132.746 | 55.588 | 87.067--194.638 |
| 远场Ag (at.%) | 0.632990 | 0.013220 | 0.621465--0.647421 |

三条路径都显示颗粒数下降、平均半径和`M6`增加、`Sv`下降，且48 h远场Ag
全部在`0.62+/-0.04 at.%`范围。β体积分数和远场Ag的重复间离散较小，
而粒子数、PSD尾部及`M6`离散很大；这正式确认后续热输运必须保留完整PSD、
`Sv`、`M6`和merge-aware来源，不能只用`Nv+mean_R`替代。

该三样本ensemble是条件路径集合，不是统计或盒子尺寸收敛。12 h三条远场
Ag均暂时高于48 h实验带，仍登记为缺少12 h实验锚点的handoff初态瞬态；
不得通过案例专用参数调整消除。PF的`Nv`仍只代表resolved beta，不与APT
全部Ag-rich对象密度一一对应。

```text
ABC_status=PASS_246CUBE_HOURLY_MERGE_DISSOLUTION_ABC_ENSEMBLE_V1
C_audit_json_sha256=7e22455465f3d8c7661673286ef9cdb47d5e1cb46ee319d3513e13d66cb5f3d6
ABC_comparator_sha256=b56f9d2daab6aeebf94e7b844a0ae1983a9b1a300adbb46ad0f71f295987ce57
ABC_comparison_audit_sha256=b4e404317388c4c2a091af002bef764f43da86bab68d1196168add209a7242ce
reports_C=reports/pf_246cube_hourly_merge_dissolution_C_v1/
reports_ABC=reports/pf_246cube_hourly_merge_dissolution_ABC_v1/
```

### 13.27 Method-1 A完整原始PSD的无位错诊断（2026-08-01）

四分之一纳米Method-1 A（job `73364`）已数值推进到最终step `152585`，
最终solver segment为PASS、stderr为空，且最大质量相对误差约
`3.27e-13`。但自动生产driver的最终身份审计为：

```text
root_status=BLOCKED_246CUBE_6H48H_PRODUCTION_DRIVER_V1
lineage_status=BLOCKED_PERIODIC_OVERLAP_PARTICLE_LINEAGE_V1
initial_particle_count=96
final_particle_count=5
unqualified_dissolution_count=60
qualified_merge_count=1
unqualified_merge_count=0
split_count=0
new_component_count=0
```

因此A没有完整production `audit.json`，不是权威生产轨迹，不能转换为
13.20的authority transport snapshot，也不能进入A/B/C ensemble。该结论
不因solver完成或总体PSD可读而改变。

用户随后要求将现有每小时总体resolved PSD仅作为非权威输运诊断输入。
远端生产根未改动；45个原始快照的`ensemble_observables.csv`和
`particle_lineage.csv`被只读复制，并在独立本地报告目录计算。该诊断只把
每一个时刻的半径多重集用于散射，不把particle ID解释为跨时刻持续身份，
也不修复或覆盖A的blocked审计。

冻结输运条件保持：

```text
A_N=1.5
dislocation_mode=DISLOCATION_OFF
S11_rate=0
S13_rate=0
yu_refit_scale_used=false
source_production_authority=false
absolute_experimental_kappa_claim=false
```

full-PSD direct sum、源PSD矩闭合、lognormal闭合和两次生成文件哈希均通过：

```text
status=PASS_PF_BLOCKED_A_HOURLY_PSD_NO_DISLOCATION_DIAGNOSTIC_V1
snapshot_count=45
source_descriptor_closure_max=8.944943409969666e-16
full_psd_vector_direct_sum_max_relative=3.0963095254558003e-16
deterministic_regeneration=PASS_IDENTICAL_FILE_SHA256
```

描述量充分性（full PSD为参考，45个快照、300--600 K、两种matrix路径）显示：

| 描述量 | fixed matrix κ MAPE | fixed Δκ NRMSE | varying matrix κ MAPE | varying Δκ NRMSE |
|---|---:|---:|---:|---:|
| `Nv+mean_R` | 0.3363% | 265.5% | 0.3376% | 50.33% |
| `Nv+mean_R+CV` lognormal | 0.04164% | 46.11% | 0.04186% | 8.759% |
| `Sv` geometric | 3.833% | 3205% | 3.853% | 608.9% |
| `Sv+M6` | 0.08265% | 48.74% | 0.08303% | 9.239% |
| full PSD | 0 | 0 | 0 | 0 |

例如300 K，6 h到48 h的full-PSD条件结果为：
`kappa_L_no_dis=2.28647 -> 2.28404 W m^-1 K^-1`（fixed 6 h matrix，
`delta_kappa_PSD=-0.002429`）；若同时采用PF far-field matrix Ag，
`kappa_L_no_dis(48h)=2.28033 W m^-1 K^-1`，
`delta_kappa_PSD_plus_matrix=-0.006144 W m^-1 K^-1`。
这些仅说明被冻结Yu散射框架下、原始A总体PSD的条件趋势，绝不构成
绝对实验热导率复现或A生产PASS。

证据：

```text
scripts/analyze_pf_blocked_a_hourly_psd_no_dislocation_diagnostic_v1.py
reports/pf_blocked_A_hourly_psd_no_dislocation_diagnostic_v1/

diagnostic_script_sha256=978bd14099522da028694e349594101735da7e78e68c35fdfb87f731f000871b
diagnostic_audit_sha256=d8ee10c6fb43afe9d7f342f8645dde3583d26710a4f8b30bb74517b8a5fa706b
full_psd_trajectory_sha256=c28e4ab885102d25bfb597e2d975493777b005ab2f5a389dc17809ef5e21955b
```

正式production ensemble仍为：

```text
PENDING_EXPERIMENT_MATRIX_ANCHORED_A_B_C_COMPLETE_PASS_AUTHORITIES
```

### 13.28 Method-1 A/B/C生产权威overlay与无位错full-PSD ensemble（2026-08-01）

13.24--13.26的三阈值小时谱系审计已明确接纳为Method-1生产的merge-aware
身份合同。原始A/B/C生产根保留其旧low-threshold tracker留下的`BLOCKED`
状态，未修改、未覆盖；而是为每个replicate建立独立authority overlay。每个
overlay同时验证：

- 原生产44个solver segment均为PASS；
- 远端44个checkpoint逐一重新读取SHA-256；
- 冻结fixture、campaign、input hash ledger和analysis binary互相一致；
- GP、GP Birth、GP release、external source和new-beta nucleation均关闭；
- 13.24--13.26小时merge-aware审计为PASS且全gate为真；
- 六个注册科学时刻`0/21798/43596/65393/108989/152585`的总体PSD与
  `Nv/Sv/M6/mean-R`重新闭合。

三条overlay均得到精确生产PASS，并且每条保留完整44-checkpoint链：

```text
PASS_246CUBE_6H48H_CONDITIONAL_PRODUCTION_V1
A_audit_sha256=dfd0ad7824fe66acd81ba0f788399a801f3055402f57d0d5f9a8ad89e9688ef4
B_audit_sha256=d0fd7e947f47935dac4a0823c05b6069e74cd0f998d1b63b3c76df47c8aeac8d
C_audit_sha256=5b22ef813ada4131b4a5760909f528c3981c65b8aaf9268b523f9e1045d2046b
```

每个replicate恰选择一个overlay authority，既有严格生产输运适配器随后
将其转换为六时刻full-PSD snapshot。最终A/B/C ensemble通过：

```text
PASS_PF_FULL_PSD_NO_DISLOCATION_TRANSPORT_ENSEMBLE_V1
A_N=1.5
dislocation_mode=DISLOCATION_OFF
S11_rate=0
S13_rate=0
yu_refit_scale_used=false
```

48 h的三样本条件ensemble（均值±样本标准差，W m^-1 K^-1）为：

| T (K) | fixed 6 h matrix kappa | PF matrix kappa | delta kappa PSD | delta kappa PSD + matrix |
|---:|---:|---:|---:|---:|
| 300 | 2.284604 ± 0.001512 | 2.279808 ± 0.006337 | -0.001867 ± 0.001507 | -0.006662 ± 0.006330 |
| 400 | 1.786014 ± 0.000623 | 1.783033 ± 0.003625 | -0.002080 ± 0.000620 | -0.005061 ± 0.003620 |
| 600 | 1.243672 ± 0.000140 | 1.242200 ± 0.001616 | -0.001518 ± 0.000139 | -0.002991 ± 0.001614 |

描述量比较以direct full PSD为权威参考；`Sv`单独的MAPE约4.44--4.46%，
`M6` Rayleigh单独约75.35--75.44%，`Nv+mean_R`约0.328--0.329%，而
`Sv+M6`矩重建约0.0896--0.0899%。因此本ensemble中`Sv`或`M6`单独均不足；
`Sv+M6`是目前最佳的压缩测试结果，但full PSD继续是正式生产输入。

该结果是“Yu公开非颗粒host + PF resolved full PSD + 无位错”的条件路径，
不是绝对实验晶格热导率复现；三样本也不等于统计/盒尺寸收敛。

证据：

```text
scripts/assemble_pf_246cube_hourly_merge_production_authority_v1.py
reports/pf_246cube_method1_production_authority_v1/

authority_assembly_script_sha256=78ed4c65603e62c5f959f05f089a4c491a968ab3b70f2d5dfaeb45174c502513
authority_selection_sha256=8574b0a0c091f936a4fdac8ab4258c9816627f690f148a74a1bdf92d2129da56
ensemble_manifest_sha256=4c726fd8aaac155d4fc746c68b8f6091bb4d31527a5ae008178a8e6eceac9cd7
ensemble_kappa_sha256=3d3df325569ba8886d15e37802afd9c1b93a6a5c873a41eabc75ff3376e14f62
ensemble_descriptor_sha256=0168864bb47db3d0fe9a2ca549710a0813f47f75d5e3f47f282c738eb19f6e45
```

### 13.29 PF无位错输运与Sheskin 6--48 h实验的定量差距（2026-08-01）

已将13.28的Method-1 A/B/C完整PSD条件输运结果与Sheskin等人2018年
PbTe--Ag2Te实验在`300 degC = 573.15 K`下的6 h和48 h热导率作边界明确的
定量比较。实验原文给出：

```text
kappa_experiment_6h_573p15K=0.85 W m^-1 K^-1
kappa_experiment_48h_573p15K=1.03 W m^-1 K^-1
delta_kappa_experiment=+0.18 W m^-1 K^-1
relative_change_experiment=+21.1765 percent
```

论文中的这两个数值是实测总热导率；作者同时报告电子热导率比晶格热导率
约低四个数量级，因此在该样品和温区可把它们作为近似晶格热导率实验锚点，
但记录中仍须保留其原始身份为`measured_total_kappa`。

当前生产输运网格包含550 K和600 K而不含573.15 K。对13.28的A/B/C均值
在550--600 K之间作线性插值，仅用于差距诊断，得到：

| 状态 | Sheskin实验 | PF输运 | PF减实验 | 相对偏高 |
|---|---:|---:|---:|---:|
| 6 h | 0.850000 | 1.300105 | +0.450105 | +52.9535% |
| 48 h，fixed 6 h matrix | 1.030000 | 1.298509 | +0.268509 | +26.0688% |
| 48 h，PF time-varying matrix | 1.030000 | 1.296904 | +0.266904 | +25.9130% |

单位均为`W m^-1 K^-1`。最完整的PF time-varying matrix路径给出：

```text
delta_kappa_PF_6h_to_48h=-0.00320088 W m^-1 K^-1
relative_change_PF=-0.246202 percent
signed_delta_kappa_gap_PF_minus_experiment=-0.183201 W m^-1 K^-1
relative_change_gap=-21.4227 percentage_points
trend_sign_match=false
```

因此当前结果在两个层面都不能宣称复现实验：绝对热导率在6 h和48 h分别
偏高约53%和26%；更重要的是，实验6--48 h增加约21.2%，而当前模型基本
不变并略微下降。13.28的PASS只证明“Yu公开非颗粒host + PF resolved beta
完整PSD + 无位错”这一条件计算链的数值和工程闭合，不是Sheskin绝对热导率
或其时间趋势的复现PASS。

当前科学解释边界冻结为：PF resolved beta人口单独产生的条件热导变化仅约
`0.1--0.3%`，不足以解释实验约21%的升高。当前实验与模型之间还存在至少
以下未闭合贡献：实验中高密度1--10 nm unresolved Ag-rich对象及其消失、
粗化过程中的应变松弛，以及当前主线明确排除的位错相关散射。不得通过
案例专用参数重调来强行消除该差距；后续若扩展，必须把resolved beta、
unresolved对象、应变和外部位错背景分通道报告。

来源与计算证据：

```text
experimental_source=Sheskin_et_al_ACS_AMI_2018_DOI_10.1021/acsami.8b15204
experimental_temperature_K=573.15
experimental_kappa_identity=MEASURED_TOTAL_KAPPA_ELECTRONIC_COMPONENT_NEGLIGIBLE
pf_source=reports/pf_246cube_method1_production_authority_v1/ensemble/ensemble_kappa_time_temperature.csv
pf_temperature_interpolation=LINEAR_550K_TO_600K_DIAGNOSTIC_ONLY
absolute_experimental_kappa_claim=false
experiment_trend_reproduction_status=FAIL_WRONG_SIGN_AND_MAGNITUDE
```

### 13.30 Method-1 A/B/C每小时完整PSD无位错Yu输运与描述量充分性（2026-08-01）

已把13.28中三个正式Method-1生产authority的全部小时快照直接接入冻结的
无位错Yu输运接口，不重跑PF，也不修改任何生产场。输入为A/B/C各45个
`6--48 h`完整连通域半径多重集，共135个真实PSD快照；合并后的连通域在
相应时刻只作为一个实际散射体计数，不重复计入其历史lineage成员。

计算合同保持不变：

```text
A_N=1.5
dislocation_mode=DISLOCATION_OFF
S11_rate=0
S13_rate=0
yu_refit_scale_used=false
temperatures_K=300,350,400,450,500,550,600
matrix_modes=fixed_6h_matrix,pf_time_varying_matrix
```

对每个快照、温度和matrix口径比较五种输入：

1. `Nv_plus_mean_R`单分散闭合；
2. `Nv_plus_mean_R_plus_CV_lognormal`固定对数正态闭合；
3. `Sv_geometric_limit`；
4. `Sv_plus_M6`两矩重建；
5. 逐颗粒直接求和的`full_PSD`权威参考。

最终共计算`3*45*7*2*5=9450`个输运单元。生产authority、原始输入哈希、
44-checkpoint链和小时merge-aware审计全部PASS；full-PSD向量/标量直接和
最大相对差为`4.775e-16`，对数正态均值/CV闭合分别为`3.468e-15`和
`9.159e-16`。六个原注册时刻与13.28的正式输运逐点复算最大相对差仅
`3.591e-16`，并保留原ID--半径配对哈希一致性。第二次独立复算的全部10个
输出文件与正式输出字节一致。

描述量误差相对于完整PSD为：

| matrix口径 | 描述量 | kappa MAPE | delta-kappa信号NRMSE | A/B/C delta严格排序一致率 |
|---|---|---:|---:|---:|
| fixed 6 h | Nv+mean R | 0.3493% | 291.30% | 31.5% |
| fixed 6 h | Nv+mean R+CV（lognormal） | 0.0371% | 42.94% | 26.0% |
| fixed 6 h | Sv | 3.8970% | 3223.11% | 45.5% |
| fixed 6 h | Sv+M6 | 0.0951% | 63.22% | 25.6% |
| PF matrix | Nv+mean R | 0.3506% | 52.64% | 44.2% |
| PF matrix | Nv+mean R+CV（lognormal） | 0.0373% | 7.77% | 70.8% |
| PF matrix | Sv | 3.9181% | 583.37% | 64.3% |
| PF matrix | Sv+M6 | 0.0955% | 11.43% | 51.0% |

因此`Nv+mean R+CV`在当前冻结的lognormal形状假设下是最佳压缩模型，优于
`Sv+M6`；但它仍不能在所有小时稳定保留A/B/C细微演化排序，尤其fixed-matrix
口径仅26.0%。小的绝对kappa误差受到共同host散射背景稀释，不能替代更严格的
delta-kappa与样本排序门。因此正式生产输运输入继续冻结为完整PSD；
`Nv+mean R+CV`只作为快速代理和敏感性描述量，不升级为权威输入。

400 K的完整PSD三样本均值显示：PF-matrix路径由6 h的`1.788093 W m^-1 K^-1`
在7 h降至`1.757263`，随后回升至48 h的`1.783033`，最终相对6 h仍为
`-0.005061 W m^-1 K^-1`。这个小时分辨结果把现有6--7 h初态/基体瞬态清楚
暴露出来，但不改变13.29的科学边界：无位错resolved-beta完整PSD路径仍未
复现实验6--48 h热导率上升趋势。

最终状态与证据：

```text
status=PASS_PF_METHOD1_ABC_HOURLY_FULL_PSD_NO_DISLOCATION_DESCRIPTOR_SUFFICIENCY_V1
source_snapshot_count=135
prediction_cell_count=9450
full_PSD_reference=true
absolute_experimental_kappa_claim=false

script_sha256=3a2b38faab689b56d216e6faaf2a29cb7bd4ed9b260c45f2f3159c7e6eef5af2
audit_sha256=588b20d1ee8a429f161bc4f0eaae0d7fab78aa389c1df25559aadac7cf6947a4
report_sha256=1acc1c3cc0f67bcc791d14c7b1666ec339bdaec3ef4842641e18642c03b4addd
predictions_sha256=79ea6c27ec901f140cc00e48582d6007b308bc02afcd3d1e61ded57844334afa
hourly_ensemble_sha256=8c09d3be8ebd7d1b8e4354e375a83f751b05791df31ee573f6d7a1ce6821676a
workbook_sha256=fce24e091f32ce8fc07593b5e2c84753928835cb395e6cbbf292efa6aaa1352e
```

报告与可复核工作簿：

```text
reports/pf_method1_abc_hourly_no_dislocation_transport_v1/
outputs/pf_method1_abc_hourly_transport_v1/pf_method1_abc_hourly_transport_summary.xlsx
```

### 13.31 Yu无位错方法对接Sheskin 2018实测热导率的正式比较合同（2026-08-02）

后续晶格热输运验证正式采用以下两层严格分离的来源身份：

1. 使用Yu 2024公开的单松弛时间Debye--Callaway方程、非颗粒host参数和
   S7--S10析出物截面，计算PF条件晶格热导率；
2. 使用Sheskin等人2018年Tailoring论文的实测热导率，作为独立外部实验
   对照，而不把Yu的计算曲线或位错关闭敏感性结果误称为实验值。

PF计算合同继续冻结为：

```text
A_N=1.5
dislocation_mode=DISLOCATION_OFF
S11_rate=0
S13_rate=0
yu_refit_scale_used=false
yu_small_big_population_replaced_by_PF_full_PSD=true
point_defect_xAg_source=PF_FAR_FIELD_MATRIX_OBSERVATION
case_specific_refit=false
```

必须在实验的相同测量温度计算。Sheskin主文Figure 5c给出AQ、6 h和48 h
的实测总热导率；Figure 7使用Wiedemann--Franz关系分离晶格和电子分量。
论文报告电子热导率比晶格热导率约低四个数量级，因此本样品和温区可将
实测总热导率作为近似晶格热导率锚点，但原始数据身份必须继续登记为
`MEASURED_TOTAL_KAPPA_ELECTRONIC_COMPONENT_NEGLIGIBLE`，不能静默改名为
直接测量的晶格热导率。

正文在`300 degC = 573.15 K`明确给出：

```text
Sheskin_measured_total_kappa_6h=0.85 W m^-1 K^-1
Sheskin_measured_total_kappa_48h=1.03 W m^-1 K^-1
Sheskin_delta_kappa_6h_to_48h=+0.18 W m^-1 K^-1
Sheskin_relative_change=+21.1765 percent
Sheskin_trend=kappa_48h_gt_kappa_6h
```

Figure 5c显示该排序不是单一端点偶然值：在论文展示的测量温区内总体保持
`kappa(48h) > kappa(AQ) > kappa(6h)`。作者将6--48 h上升归因于析出物
粗化和数密度降低使声子散射减弱，同时伴随基体应变松弛。这里的“高/低”
只描述热导率方向；不得把48 h称为热电性能必然“更好”，因为较低晶格
热导率通常更有利于热电优值。

使用512点Gauss--Legendre在573.15 K直接复算13.28的Method-1 A/B/C
PF-matrix full-PSD ensemble，得到：

```text
PF_no_dis_kappa_6h_573p15K=1.2980916621210676 W m^-1 K^-1
PF_no_dis_kappa_48h_573p15K=1.2948991542326225 W m^-1 K^-1
PF_delta_kappa_6h_to_48h=-0.003192507888445162 W m^-1 K^-1
PF_relative_change=-0.24593855592821842 percent
PF_trend_sign_match=false
PF_6h_relative_above_Sheskin=52.716666131890314 percent
PF_48h_relative_above_Sheskin=25.718364488604124 percent
```

这些直接温度点结果替代13.29中的550--600 K线性插值作为573.15 K的更精确
诊断，但不覆盖或删除13.29的历史插值证据。当前resolved-beta无位错路径
依然不能宣称复现Sheskin的绝对值或6--48 h增幅。

Yu 48 h公开参数关闭位错时的结果必须单独登记为模型敏感性基准：

```text
Yu_48h_no_dis_kappa_303p94K=2.4532892127605708 W m^-1 K^-1
Yu_48h_no_dis_kappa_573p15K=1.3589214073546045 W m^-1 K^-1
identity=YU_PUBLISHED_PARAMETER_DISLOCATION_OFF_SENSITIVITY_NOT_EXPERIMENT
```

当前PF 48 h在573.15 K比Yu无位错敏感性基准低约`4.7113%`，可称为数值上
接近该条件模型基准；但这一接近不能替代与Sheskin实验`1.03 W m^-1 K^-1`
的比较，也不能作为实验复现PASS。Yu与Sheskin样品名义组成都为
`(PbTe)0.97(Ag2Te)0.03`，但论文、制样和注册微观结构参数不完全相同，因此
跨论文对照属于外部验证，不是相同试样的逐点同一性证明。

后续正式比较必须输出：

- 与Sheskin相同温度点的6 h和48 h `kappa_L(T)`；
- 每个状态的绝对误差、相对误差和全温区MAPE；
- 每个温度的`kappa_48h > kappa_6h`趋势门；
- 573.15 K的绝对端点、`delta_kappa`和相对增幅；
- PF完整PSD、远场Ag、host参数、求积器和实验数字化provenance；
- measured total、Wiedemann--Franz派生lattice以及Yu no-dis sensitivity三种
  数据身份分列报告。

除正文明确给出的573.15 K端点外，主文其余温度值只有图形。全温区定量
比较应优先取得作者原始数据或Supporting Information；若只能数字化
Figure 5c/Figure 7，必须保存像素、坐标标定、曲线身份和数字化不确定度，
不得把数字化点冒充表格原值。

当前科学判定冻结为：

```text
yu_no_dis_pf_interface_status=PASS_NUMERICAL_AND_ENGINEERING
sheskin_external_comparison_status=FEASIBLE_AND_REQUIRED
current_absolute_experiment_status=FAIL_HIGH_BIAS
current_6h_48h_trend_status=FAIL_WRONG_SIGN_NEAR_FLAT
resolved_beta_only_explains_21percent_increase=false
absolute_experimental_kappa_claim=false
```

当前差距可用于约束未闭合通道，包括实验中的高密度unresolved Ag-rich对象
及其消失、非位错的相干/基体应变散射与松弛，以及外部位错背景；不得为匹配
Sheskin端点而对Yu host、PF物理参数或单个案例进行专用重调。任何新增通道
必须与resolved beta、point defect和external dislocation分通道报告，并先
通过库存、单位、确定性和敏感性验收。

### 13.32 全局resolved-PSD密度对比与检查点应变场审计（2026-08-02）

只读审计
[`global_resolved_psd_strain_audit_v1`](reports/global_resolved_psd_strain_audit_v1/)
在冻结的 Yu S7--S10 density-contrast、固定 Method-1 β库存和
`xAg=0.0062` 下得到：

```text
density_only_global_status=NO_GO_RESOLVED_DENSITY_CONTRAST_GLOBAL_RANGE
resolved_radius_scan_min_R_nm=20.5
resolved_radius_scan_min_kappa=1.2960071108433275 W m^-1 K^-1
maximum_positive_delta_kappa=0.019267359064296663 W m^-1 K^-1
maximum_relative_increase_percent=1.4847411907303796
any_resolved_endpoint_pair_within_5percent=false
```

这排除了仅通过改变相同resolved β库存的数密度、平均半径、lognormal宽度或
已测双峰PSD来解释实验`+0.18 W m^-1 K^-1`趋势的路线；不是对未解析
Ag-rich对象、应变散射或外部位错通道的否定。

同一审计检查A/B/C不可变V4 checkpoint后，拒绝了不合法的时间层混合：其中
位移warm state属于源场`n-1`，但只有接受后的`phi/Y/xB(n)`被保存，端点没有
匹配的`n-1`场，6 h fixture也没有已组装的运行时位移场。因此应变方差、频谱
与能量闭合不能精确重建，状态为
`INCONCLUSIVE_STRAIN_FIELDS_NOT_RECOVERABLE`，而不是“应变影响很弱”。在另行
授权且checkpoint设计保存source-time-level一致的弹性快照前，
`R1_R2_dynamic_PF_status=DEFERRED`。两次隔离的全局分析33个文件逐字节一致。

### 13.33 400-cube fixed-总库存 elastic PSD-sign pilot 准备已提交（2026-08-02）

按用户授权，已启动一条独立的、400 nm 周期、全弹性、
`xB_total=0.03`、窄 PSD 的“正热导增量可行性 pilot”。它不替代原有
246-cube 历史证据，不是 box-converged 或 ensemble-qualified 结论，也不可声称复现
实验绝对热导率。GP、外部源和新粒子插入保持关闭。

之前的 96-cube 原生 R17 有限窗口在不改变任何物理参数、
质量合同或接受阈值的前提下，无法在规定迭代上限内闭合能量平台；
没有产生 fixture 或生产输出。独立的 160-cube 原生 R17 探针 job `73796`
已通过：`converged_iter=31214`、`rms_res=4.28760794e-4`、
`energy_diff_rel=4.24444610e-9`、体积误差 `1.81580994e-11`、质量误差
`1.57862148e-16`。这只改变了轮廓的数值隔离窗口，不是物理 retune，不使用
96-cube 失败场。

后续于 `2026-08-02T06:26:26Z` 提交 V3 全库准备 job `73798`：

```text
source=/data/home/luozhiheng/tmp/codex_pf_400cube_fixed_xb03_elastic_psd_sign_v3_20260802
preparation_root=/data/home/luozhiheng/tmp/pf_400cube_fixed_xb03_elastic_psd_sign_prepare_v3_20260802
reserved_production_root=/data/home/luozhiheng/tmp/pf_400cube_6h48h_fixed_xb03_elastic_psd_sign_pilot_v3_20260802
queue=gpu_uvip/gpu_uvip; resources=1 GPU, 4 CPUs, mem=0
profile_grid=160^3; radii_nm=17.0--20.0 by 0.5 nm
fixture_spec_sha256=a87daffc3816bebcff7c8735aace3e00ab40100649383fe4f7a3c2c958994410
preparation_sbatch_sha256=8fa8f3bea1ddca0dba60451d544ec1cc8dc2a378d4549a5ffef3910dd551f4bd
```

V3 必须依次通过七个原生全弹性 profile、400-cube fixture 结构/库存/周期
分离审计以及生产二进制的 zero-macrostep 预检。只有全部 PASS 才会自动提交独立
48 h 生产 job（`dt_code=0.02`、`final_step=152585`、48 h、禁止 bulk VTK）。在那之前
不存在可登记的生产 job ID、fixture/binary/parameter hash 或 checkpoint 哈希链。更新登记见
`reports/pf_400cube_fixed_xb03_elastic_psd_sign_pilot_v1/submission_registry_v3.md`。

### 13.34 Sheskin AQ背景 + PF resolved界面/相干应变盲预测（2026-08-02）

正式报告：
[`sheskin_pf_interface_strain_blind_prediction_v1`](reports/sheskin_pf_interface_strain_blind_prediction_v1/)。
终态：

```text
FAIL_RESOLVED_ONLY_AFTER_INTERFACE_AND_STRAIN_TEST
```

本阶段没有修改A/B/C生产场、原始checkpoint、PF热力学、初始PSD、eigenstrain、
弹性常数或Yu host参数；没有使用Yu `s_dis=0.1172768`，没有48 h拟合，
没有commit或push。Stage-0冻结density-only基线在573.15 K逐字复现：

```text
kappa_6h  = 1.2980916621210676 W m^-1 K^-1
kappa_48h = 1.2948991542326225 W m^-1 K^-1
```

AQ只用于选择时间不变背景。18个Sheskin source-literal AQ结构包络均选择
`H2: tau_bg^-1=A2*omega^2`，`A2=1.8410661899e-15`--
`2.8228972476e-15 s`，AQ MAPE约4.05--4.15%；但所有单参数候选的绝对
chi-square拟合都差，故该项只能称为经验背景包络，不得命名为具体缺陷。
Figure 5c为数字化的measured-total kappa，和WF-derived lattice kappa身份保持分离。

新增只读`MECHANICS_ONLY_ACCEPTED_FIELD_REPLAY_V1`在workstation资格化。同步短案例中
checkpoint-warm replay与online accepted-field参考的source/位移/应变/应力/能量逐字节
一致，能量、应变L2和应力L2误差均为0；zero-initialized sensitivity虽满足能量闭合，
但应变/应力L2超过预注册门，因此登记`REJECTED_NOT_QUALIFIED`，不得用于authority。
A/B/C的6、12、18、24、36、48 h共18个场均从精确fixture/权威checkpoint只读重求解，
最大残差`9.1125824426e-7`，不推进PF时间、不写checkpoint。

周期marching-cubes、周期颗粒/法向/曲率、距离壳层、accepted-field应变/应力和3D FFT
分析两次逐字节一致。A/B/C的48/6 h平均比例为：

```text
Sv                              = 0.39530647550891224
hydrostatic total variance      = 1.283362970311322
deviatoric equivalent variance  = 1.1967831639805402
hydrostatic mid-q power         = 0.3062474378035795
hydrostatic high-q power        = 0.4583384203816745
```

因此界面面积为`STRONG_LEVERAGE`；总应变方差虽上升，但mid/high-q显著下降，登记
`SPECTRAL_REDISTRIBUTION_LEVERAGE`。不得用平均弹性能或总应变方差替代正式频谱模型。

两个正式动态候选均冻结文献与单位合同：

1. MI使用Hanus/Dames-Chen频率依赖transmissivity及`n_I=Sv/2`立体学映射，
   `tau_I^-1=(2/3)v Sv alpha omega/omega_D`；每个AQ背景成员只用6 h全温区、
   A/B/C共同拟合一个`alpha`。16/18成员在预注册`[0.1,10]`内，两个极端50 nm
   球上界诊断成员达到上界并拒绝物理性。
2. MS使用无自由参数的scalar Born trace-strain频谱积分，
   `tau_S^-1=gamma^2 v/(4pi) integral q^3 S_trace(q)dq`，`gamma=1.96`，
   零模移除，`q>pi/dx`不外推；Yu S12/S13位错模型不使用。

冻结manifest后才执行48 h比较。MI在全部18个背景成员、全部注册温区均给出正趋势，
但573.15 K的背景成员恢复范围仅`+2.1148%`--`+7.2685%`，48 h预测范围
`0.83637`--`0.91141 W m^-1 K^-1`，未达到预注册10--30%恢复门，也未进入
实验`1.03 W m^-1 K^-1`的10%端点门。MS保持约`-0.18%`趋势且散射率远小于
host/background；MIS与MI近乎相同。故resolved界面/相干应变虽然结构杠杆强，
在有来源、单位闭合的公式中仍不足以解释实验`+0.18 W m^-1 K^-1`恢复。

不得通过重调PF PSD、Yu `A_N`、eigenstrain或植入位错比例补偿该失败。后续若继续，
必须建立独立约束的新物理合同，例如polarization-resolved PbTe/Ag2Te界面传输或有
独立实验身份的未解析缺陷通道；不得用本次48 h端点反向拟合。冻结manifest SHA-256为
`8e7dfa0259830ab73c81ef10410c3505b85a0cf81d7d3e27ab42c94562716234`，
completion audit SHA-256为
`05cf158dfb72775b09da3810282018576983cfd3dfaa7bd507179fe4904d8421`。

### 13.35 400-cube fixed-总库存 elastic PSD-sign pilot — V3/V4修复与V5静态准备（2026-08-02）

V3 已完成全部七条 `17.0--20.0 nm`、`160^3` 原生全弹性 profile，R20 的最终
审计为 `PASS`（`converged_iter=36812`、`energy_diff_rel=1.73219609e-9`、体积误差
`1.58448873e-11`、质量误差 `3.70155512e-16`），profile-library manifest SHA-256为
`a1b0dba740113cdfa97cd86c762030fe190fe23036487a82afff6df7770118da`。

V3 在 fixture 组装前失败，原因是通用 library verifier 遗留了仅允许 `96^3` 的
旧数值窗口检查，拒绝了本次规格明确要求的 `160^3` 原生库。V4 修复该尺寸合同后，
又暴露出组装器把 pure-beta 核心的 `alpha=1-h=0` 误认为非法 matrix fraction。原始
profile 中该区域的 `delta_C_relaxation` 最大仅 `6.279385872517745e-17`，是浮点残差；
没有物理库存可以或应当由 matrix composition 承载。

V5 仅修复上述数值组装合同：由 fixture spec 显式传入 native shape；(h(\phi))采用
端点稳定但解析等价的多项式求值；仅在 `alpha>1e-12` 的 matrix-support 内计算
`delta_C/alpha`，并以 `1e-14` 门禁止 pure-beta 核心出现任何可分辨的 matrix excess。
它不缩放、插值、剪裁或改变 profile、PSD、总库存、热力学、迁移率、界面、弹性、dt或
时间映射。旧96-cube fixture 的19项兼容测试继续全通过。

`2026-08-02` 已提交唯一 V5 静态准备 job `73849`：

```text
source=/data/home/luozhiheng/tmp/codex_pf_400cube_fixed_xb03_elastic_psd_sign_v5_20260802
preparation_root=/data/home/luozhiheng/tmp/pf_400cube_fixed_xb03_elastic_psd_sign_prepare_v5_20260802
reserved_production_root=/data/home/luozhiheng/tmp/pf_400cube_6h48h_fixed_xb03_elastic_psd_sign_pilot_v5_20260802
queue=gpu_uvip/gpu_uvip; resources=1 GPU, 4 CPUs, mem=0
fixture_spec_sha256=a87daffc3816bebcff7c8735aace3e00ab40100649383fe4f7a3c2c958994410
reused_profile_library=V3 hash-pinned 160^3 17--20 nm library
```

V5 已精确通过 fixture、静态库存/远场/周期分离审计和zero-macrostep production
preflight，并自动提交独立48 h生产 job `73850`（`RUNNING`）及其只读tail job `73851`
（`afterany:73850`）。登记如下：

```text
output_root=/data/home/luozhiheng/tmp/pf_400cube_6h48h_fixed_xb03_elastic_psd_sign_pilot_v5_20260802
fixture_sha256=b8b59357b659078e51077330f4fe1c31caf302c2605e809c71d8e41b18771b13
profile_library_sha256=a1b0dba740113cdfa97cd86c762030fe190fe23036487a82afff6df7770118da
parameter_sha256=3571c917cb5696f85ff5a2dbe9d43e7c7ed6c1634c1b13f9a2777a0b2aebadc3
binary_sha256=759956a89780db9a19ccd51b4115319de47463a3fd7b8d446a022a74c9beeb3b
production_sbatch_sha256=ea490d4f6bcb17ab9970dd84ea3dba0a71981a3e4dbb499a861b106e08c146b0
checkpoint_schedule_sha256=fc593341b62d806d069066261baf8d18cbb58ce97538b6c8cad1ba29d2e7081d
```

首个注册的一小时 checkpoint（step `3633`）已完成：segment status、零模和弹性
warm-start-residual审计均为 `PASS`，总质量为精确的
`1.92000000000000000e+06` code units（reported mean mass error `0`），
checkpoint SHA-256为
`c32e297aafd5c79a52f4ee2831a47c77e6aef4887546985be0537438628374ff`。
该段用时 `1274.271 s`（约`0.3507 s/step`），无stderr；生产已从该完整checkpoint
自动继续，尚不可据此推断48 h热输运趋势。

第二个逐小时checkpoint（step `7266`）随后也以独立segment status、零模和弹性
审计通过：末态总质量为`1.92000000000001490e+06` code units，reported mean mass
error为`2.32830643653869618e-16`，last zero-mode lambda为`0`；弹性求解器保持
`0`个未收敛步、平均`2.000000000`次迭代。checkpoint SHA-256为
`18069e50b75d071d91dda4462c83d98f56f5973f67a955e3fd22296d57a34b6c`，segment与
driver stderr仍为空。该运行仍在继续，以上只构成早期数值完整性证据，不构成48 h
微观结构或热输运结论。

第三个逐小时checkpoint（step `10899`）同样通过：末态总质量为
`1.91999999999999953e+06` code units，reported mean mass error为
`-7.27595761418342557e-18`，last zero-mode lambda为`9.99323722599212514e-12`；
全弹性求解器继续为`0`个未收敛步，平均`2.785026149`次迭代，最后残差
`3.81400348790993147e-07`。checkpoint SHA-256为
`987d5d84f7a30be4ef5a4ee8e5ed3c470665ea83b705a2e43da899ac33721ce2`，segment与
driver stderr仍为空。生产已从此点继续到step `14532`；这些是早期数值完整性证据，
仍不得据此给出48 h结构或热输运结论。

第四个逐小时checkpoint（step `14532`）也通过：末态总质量精确为
`1.92000000000000000e+06` code units、reported mean mass error为`0`，last
zero-mode lambda为`2.29779068994222925e-10`。全弹性求解器仍为`0`个未收敛步，
平均`3.000000000`次迭代、最后残差`3.28219959891018171e-07`。checkpoint SHA-256为
`e3a08ae6545d82a3bef55dbd6c99d8b5712b75f741e753be6bcda03494f07229`，segment与
driver stderr为空；生产已从此完整checkpoint继续至step `18165`。这仍是早期数值
完整性证据，而非48 h结构或热输运结论。

第五个逐小时checkpoint（step `18165`）随后通过并恢复至`step 21798`：末态总质量为
`1.91999999999999977e+06` code units，reported mean mass error为
`-3.63797880709171279e-18`，last zero-mode lambda为`1.38992016306014153e-10`；
checkpoint/restart provenance为`RESTORED_AND_VALIDATED`。全弹性求解器继续为`0`个
未收敛步，平均`3.000000000`次迭代、最后残差`3.27212251618144464e-07`。checkpoint
SHA-256为`94eacf290d9a97f165f762cbc00af4d4fb4ba47e12552dc4c719f0a576d9da9b`；该段
用时`1324.405 s`（`0.360767 s/step`）、报告峰值显存`17.69 GB`，segment和随后
step `21798`的stderr均为空。日志中的VTK路径均为明确的suppression notice，未写出
bulk VTK。以上仍仅构成早期数值完整性证据，不构成48 h微观结构或热输运结论。

第六个逐小时checkpoint（step `21798`，约12 h）也独立通过：末态总质量为
`1.91999999999999977e+06` code units，reported mean mass error为
`-3.63797880709171279e-18`，last zero-mode lambda为
`1.75008487189232911e-11`，并以`RESTORED_AND_VALIDATED` checkpoint provenance
继续。全弹性求解器保持0个未收敛步、3次迭代、最后残差
`3.54327388899547215e-07`。checkpoint SHA-256为
`b9cf2c34441353ca7d6d1697f3709d619ce3bb28d3c53625b34e9cc4740e0469`；该段
用时`1325.365 s`，segment和driver stderr仍为空。driver已继续进入step
`25431`段。以上仍只是逐段数值完整性证据，未形成48 h微观结构或热输运结论。

第七个逐小时checkpoint（step `25431`，约13 h）也独立通过：末态总质量精确为
`1.92000000000000000e+06` code units，reported mean mass error为`0`，last
zero-mode lambda为`-5.85924929899351371e-11`，并以`RESTORED_AND_VALIDATED`
checkpoint provenance继续。全弹性求解器保持0个未收敛步、3次迭代、最后残差
`3.91408931778244418e-07`。checkpoint SHA-256为
`ea7ebabb71804fca039a066c527e2e2d19838b7c502ef09b9ee2c919f807f54c`；该段
用时`1325.970 s`（`0.361207 s/step`）、报告峰值显存`17.69 GB`，segment和
driver stderr仍为空。日志中的VTK路径均为明确的suppression notice，未写出bulk
VTK。driver已从该完整checkpoint继续；这些仍只是逐段数值完整性证据，尚未形成
48 h微观结构或热输运结论。

第八个逐小时checkpoint（step `29064`，约14 h）也独立通过：末态总质量精确为
`1.92000000000000000e+06` code units、reported mean mass error为`0`，last
zero-mode lambda为`-2.72652403729504141e-11`，并以`RESTORED_AND_VALIDATED`
checkpoint provenance继续。全弹性求解器保持0个未收敛步、3次迭代、最后残差
`4.55288361710837811e-07`。checkpoint SHA-256为
`91e1029a2130986d6a8624730052aae442887efce345eb9c42ac1dabbb52f3e6`；该段
用时`1309.685 s`（`0.360497 s/step`）、报告峰值显存`17.69 GB`，segment和
driver stderr仍为空。日志中的VTK路径均为明确的suppression notice，未写出
bulk VTK。driver已从该完整checkpoint继续；这些仍只是逐段数值完整性证据，
尚未形成48 h微观结构或热输运结论。

第九个逐小时checkpoint（step `32697`，约15 h）也独立通过：末态总质量精确为
`1.92000000000000000e+06` code units、reported mean mass error为`0`，last
zero-mode lambda为`2.28253918187494044e-09`，并以`RESTORED_AND_VALIDATED`
checkpoint provenance继续。全弹性求解器保持0个未收敛步、3次迭代、最后残差
`6.87372362654252257e-07`。checkpoint SHA-256为
`e9157117506a04b7880ecdcbfb00a4c137f33ad8fcf29c8ed4f42b02a20eea39`；该段
用时`1310.813 s`（`0.360807 s/step`）、报告峰值显存`17.69 GB`，segment和
driver stderr仍为空。日志中的VTK路径均为明确的suppression notice，未写出
bulk VTK。driver已从该完整checkpoint继续；这些仍只是逐段数值完整性证据，
尚未形成48 h微观结构或热输运结论。

第十个逐小时checkpoint（step `36330`，约16 h）也独立通过：末态总质量精确为
`1.92000000000000000e+06` code units、reported mean mass error为`0`，last
zero-mode lambda为`1.58670315016957542e-09`，并以`RESTORED_AND_VALIDATED`
checkpoint provenance继续。全弹性求解器保持0个未收敛步、3次迭代、最后残差
`4.26463449116233394e-07`。checkpoint SHA-256为
`911cd636b32f1c41ce84d73a90c63640b784a4bc8778b9d3d353f9eaec01ce92`；该段
用时`1311.588 s`（`0.361021 s/step`）、报告峰值显存`17.69 GB`，segment和
driver stderr仍为空。日志中的VTK路径均为明确的suppression notice，未写出
bulk VTK。driver已从该完整checkpoint继续；这些仍只是逐段数值完整性证据，
尚未形成48 h微观结构或热输运结论。

第十一个逐小时checkpoint（step `39963`，约17 h）也独立通过：末态总质量精确为
`1.92000000000000000e+06` code units、reported mean mass error为`0`，last
zero-mode lambda为`-9.75207175307529340e-12`，并以`RESTORED_AND_VALIDATED`
checkpoint provenance继续。全弹性求解器保持0个未收敛步、3次迭代、最后残差
`4.79256439537517968e-07`。checkpoint SHA-256为
`814138364b8baf3d26adf57fc0ec9b7927aaa20a0ff15577fee4e485937e4272`；该段
用时`1324.290 s`（`0.364517 s/step`），segment和driver stderr仍为空。driver已从该
完整checkpoint继续；这些仍只是逐段数值完整性证据，尚未形成48 h微观结构或热输运结论。

第十二个逐小时checkpoint（step `43596`，也是冻结的18 h科学端点）也独立通过：末态
总质量为`1.91999999999999953e+06` code units、reported mean mass error为
`-7.27595761418342557e-18`，last zero-mode lambda为`9.20093789439046313e-12`，
并以`RESTORED_AND_VALIDATED` checkpoint provenance继续。全弹性求解器保持0个未收敛
步、3次迭代、最后残差`5.47462523155116350e-07`。checkpoint SHA-256为
`27d2801f67f5809ee09320e1b8e6c29da941d5da977efc6ddddaf28dc53cb757`；该段用时
`1325.068 s`（`0.364731 s/step`），segment和driver stderr仍为空。driver已从该
完整checkpoint继续；这些仍只是逐段数值完整性证据，尚未形成48 h微观结构或热输运结论。

第十三个逐小时checkpoint（step `47229`，约19 h）也独立通过：末态总质量精确为
`1.92000000000000000e+06` code units、reported mean mass error为`0`，last
zero-mode lambda为`1.28078347919849651e-10`，并以`RESTORED_AND_VALIDATED`
checkpoint provenance继续。全弹性求解器保持0个未收敛步、3次迭代、最后残差
`5.74291806681941649e-07`。checkpoint SHA-256为
`b9ae63d75a77991079432f248e9657e9a079a8d04e232d71a56f7854962648f7`；该段
用时`1325.555 s`（`0.364865 s/step`），segment和driver stderr仍为空。driver已从该
完整checkpoint继续；这些仍只是逐段数值完整性证据，尚未形成48 h微观结构或热输运结论。

第十四个逐小时checkpoint（step `50862`，约20 h）也独立通过：末态总质量精确为
`1.92000000000000000e+06` code units、reported mean mass error为`0`，last
zero-mode lambda为`1.47022243821484707e-09`，并以`RESTORED_AND_VALIDATED`
checkpoint provenance继续。全弹性求解器保持0个未收敛步、3次迭代、最后残差
`6.27125232078676278e-07`。checkpoint SHA-256为
`a28af8658b5b0c3207de96c542355c71d8177b21bc837ff3b02b52a38609d8e3`；该段
用时`1323.924 s`（`0.364416 s/step`），segment和driver stderr仍为空。driver已从该
完整checkpoint继续；这些仍只是逐段数值完整性证据，尚未形成48 h微观结构或热输运结论。

第十五个逐小时checkpoint（step `54495`，约21 h）也独立通过：末态总质量精确为
`1.92000000000000000e+06` code units、reported mean mass error为`0`，last
zero-mode lambda为`8.28397007500473634e-09`，并以`RESTORED_AND_VALIDATED`
checkpoint provenance继续。全弹性求解器保持0个未收敛步、3次迭代、最后残差
`9.44985475096211181e-07`。checkpoint SHA-256为
`08480df60aa4351b5f2c0dba0f4d58c73ad0dfa1d1f197fb44482232592f44db`；该段
用时`1324.523 s`（`0.364581 s/step`），segment和driver stderr仍为空。driver已从该
完整checkpoint继续；这些仍只是逐段数值完整性证据，尚未形成48 h微观结构或热输运结论。

第十六个逐小时checkpoint（step `58128`，约22 h）也独立通过：末态总质量精确为
`1.92000000000000000e+06` code units、reported mean mass error为`0`，last
zero-mode lambda为`4.16513251794058427e-09`，并以`RESTORED_AND_VALIDATED`
checkpoint provenance继续。全弹性求解器保持0个未收敛步，平均`3.019267823`次迭代、
最后残差`6.94076797800361597e-07`。checkpoint SHA-256为
`e3fc1b441116026be2c221947a785555802344bbe6833691d54b5809bb10630a`；该段用时
`1329.691 s`（`0.366004 s/step`），segment和driver stderr仍为空。driver已从该
完整checkpoint继续；这些仍只是逐段数值完整性证据，尚未形成48 h微观结构或热输运结论。

第十七个逐小时checkpoint（step `61761`，约23 h）也独立通过：末态总质量为
`1.92000000000000023e+06` code units、reported mean mass error为
`3.63797880709171279e-18`，last zero-mode lambda为`2.07510872612288599e-09`，
并以`RESTORED_AND_VALIDATED` checkpoint provenance继续。全弹性求解器保持0个未收敛
步，平均`3.005780347`次迭代、最后残差`7.90848047465642520e-07`。checkpoint SHA-256为
`c33f00d145b5f68e304a725baa8e7b5ae900b7376cc461ed2c3706587116ad7e`；该段用时
`1325.222 s`（`0.364773 s/step`），segment和driver stderr仍为空。

第十八个checkpoint为冻结的精确24 h科学端点（step `65393`），也独立通过：末态总质量
精确为`1.92000000000000000e+06` code units、reported mean mass error为`0`，last
zero-mode lambda为`3.64914014783214586e-09`，并以`RESTORED_AND_VALIDATED`
checkpoint provenance继续。全弹性求解器保持0个未收敛步，平均`3.032764317`次迭代、
最后残差`6.53873172380587027e-07`。checkpoint SHA-256为
`217f1b0130e136f70b02e154723752bff6987ec91e28c8abcebdf1821a244760`；该段用时
`1328.986 s`（`0.365910 s/step`），segment和driver stderr仍为空。

第十九个checkpoint（step `65394`）保存紧邻24 h端点的普通逐小时链，同样通过：总质量
精确、last zero-mode lambda为`3.89509725248233663e-09`、全弹性求解器0个未收敛步，
并以`RESTORED_AND_VALIDATED` provenance继续。checkpoint SHA-256为
`48c1d0087da4d42d911529b65a21aae8999c55ae252dfaa93423e5f44383b497`；该1步衔接段用时
`22.545 s`，segment和driver stderr仍为空。上述仅为逐段数值完整性证据，尚未形成48 h
微观结构或热输运结论。

第二十个逐小时checkpoint（step `69027`，约25 h）也独立通过：末态总质量为
`1.92000000000000047e+06` code units、reported mean mass error为
`7.27595761418342557e-18`，last zero-mode lambda为`6.93841951952144339e-10`，
并以`RESTORED_AND_VALIDATED` checkpoint provenance继续。全弹性求解器保持0个未收敛
步、3次迭代、最后残差`6.35028316753186218e-07`。checkpoint SHA-256为
`1f8c671199948889a2475c4e3315a4cb85982e04088b75f40e16a284c43b390d`；该段用时
`1326.793 s`（`0.365206 s/step`），segment和driver stderr仍为空。driver已从该
完整checkpoint继续；这些仍只是逐段数值完整性证据，尚未形成48 h微观结构或热输运结论。

第二十一个逐小时checkpoint（step `72660`，约26 h）也独立通过：末态总质量为
`1.91999999999999977e+06` code units、reported mean mass error为
`-3.63797880709171279e-18`，last zero-mode lambda为`1.18275928031617289e-09`，
并以`RESTORED_AND_VALIDATED` checkpoint provenance继续。全弹性求解器保持0个未收敛
步，平均`3.004954583`次迭代、最后残差`6.17810852028692624e-07`。checkpoint SHA-256为
`0b6ab43a9dc73ba88e0cebdfc8b7dc9a4291f43a858b03ebc715eab430adc002`；该段用时
`1328.840 s`（`0.365769 s/step`），segment和driver stderr仍为空。driver已从该
完整checkpoint继续；这些仍只是逐段数值完整性证据，尚未形成48 h微观结构或热输运结论。

第二十二个逐小时checkpoint（step `76293`，约27 h）也独立通过：末态总质量为
`1.91999999999999977e+06` code units、reported mean mass error为
`-3.63797880709171279e-18`，last zero-mode lambda为`1.44763867299962247e-09`，
并以`RESTORED_AND_VALIDATED` checkpoint provenance继续。全弹性求解器保持0个未收敛
步、3次迭代、最后残差`6.13050710389133641e-07`。checkpoint SHA-256为
`c1201275d1fb020dbab37963e8ea4c3abe338fb71413a8c48c6e08f9765c1d91`；该段用时
`1327.788 s`（`0.365480 s/step`），segment和driver stderr仍为空。driver已从该
完整checkpoint继续；这些仍只是逐段数值完整性证据，尚未形成48 h微观结构或热输运结论。

第二十三至第二十五个逐小时checkpoint（step `79926`、`83559`、`87192`，约28–30 h）
均独立通过。三段的checkpoint SHA-256依次为
`afe05035e74953697f29f3334f9201e848d9dfb3fa5b11684ac31a34892da83f`、
`7980624a94467d376d49ae66376dddaa1993b9a2b30f2cfcd950f49c5b7342d2`和
`50920ca58382bd032a193b3c5dcadbf7b218a0c74534af73721d4d7090488d06`。总质量误差
分别为`0`、`0`和`-3.63797880709171279e-18`，last zero-mode lambda分别为
`9.88462230709040779e-09`、`9.75214895656759957e-09`和`7.77750696572297590e-10`；
均为`RESTORED_AND_VALIDATED` provenance、0个弹性未收敛步，最后残差分别为
`4.70420898561883463e-07`、`6.74480988648326349e-07`和`5.58767342317890684e-07`。
三段耗时分别为`1326.497 s`、`1326.111 s`和`1326.343 s`（约`0.365 s/step`），
segment和driver stderr均为空。上述仅为逐段数值完整性证据，尚未形成48 h微观结构或
热输运结论。

第二十六个逐小时checkpoint（step `90825`，约31 h）也独立通过：末态总质量精确为
`1.92000000000000000e+06` code units、reported mean mass error为`0`，last
zero-mode lambda为`1.30183524411109767e-08`，并以`RESTORED_AND_VALIDATED`
checkpoint provenance继续。全弹性求解器保持0个未收敛步、3次迭代、最后残差
`5.57415815526050696e-07`。checkpoint SHA-256为
`b07ac10e536ea205015573dca20282585c55d7aff30a1fe80d96321ce0f6af6a`；该段用时
`1326.513 s`（`0.365129 s/step`），segment和driver stderr仍为空。driver已从该
完整checkpoint继续；这些仍只是逐段数值完整性证据，尚未形成48 h微观结构或热输运结论。

第二十七个逐小时checkpoint（step `94458`，约32 h）也独立通过：末态总质量精确为
`1.92000000000000000e+06` code units、reported mean mass error为`0`，last
zero-mode lambda为`3.56625738490573437e-09`，并以`RESTORED_AND_VALIDATED`
checkpoint provenance继续。全弹性求解器保持0个未收敛步、3次迭代、最后残差
`2.72329941864401677e-07`。checkpoint SHA-256为
`8b6c40dfa6c3872acbfa572c32bb660e5d2b40bd9ca6765beaf3a5c939bc3c7b`；该段用时
`1324.916 s`（`0.364689 s/step`），segment和driver stderr仍为空。driver已从该
完整checkpoint继续；这些仍只是逐段数值完整性证据，尚未形成48 h微观结构或热输运结论。

生产运行期间做了一项只读分析合同修复，不修改`main_cuda`、参数、fixture、checkpoint或
PF路径：原 completion audit 错把checkpoint schedule限制为纯`3633`步倍数，遗漏了
冻结campaign同时要求保存的`65393`（24 h）和`108989`（36 h）精确科学端点；每个端点
相邻的`65394`与`108990`一小时checkpoint仍同时保留。修复后审计从campaign manifest
构造完整44项 hourly/science chain，并通过端点回归测试。更新分析脚本SHA-256为
`bb09e883cf50d44090b16dda4320c2927d11dd0091eb6a766816b4530f17a051`，将由生产末端
`analysis_hashes.sha256`记录。

V3/V4没有生产job，且其输出均作为失败诊断证据保留。详情见
`reports/pf_400cube_fixed_xb03_elastic_psd_sign_pilot_v1/submission_registry_v5.md`。

### 13.36 V3绝对端点可达性与V2背景可转移性审计（2026-08-02）

独立只读输运审计位于
`reports/v3_absolute_endpoint_feasibility_audit_v1/`。本审计没有启动或修改PF、DFT、
AIMD、TDEP或MACE任务，没有修改A/B/C权威轨迹、PSD、V1/V2冻结目录和现有V3报告；
保持`A_N=1.5`、所有位错散射为0，未使用Yu的`0.1172768/0.119`缩放，也未使用负散射率。

冻结V1/V2再次通过只读复现：V1为
`1.2980916621210676 -> 1.2948991542326225 W m^-1 K^-1`，V2 M0为
`0.8733452621717084 -> 0.8716818275173842`，V2 MIS为
`0.8208832450913344 -> 0.8490371923264635`（6 h -> 48 h，573.15 K）。因此若原样
继承V2的AQ经验背景A2，48 h零动态散射上限已经比实验`1.03`低
`0.158318172482616 W m^-1 K^-1`；从实验6 h的`0.85`出发最多只能恢复
`0.021681827517384 W m^-1 K^-1`，即`2.550803237%`。该分支严格冻结为
`NO_GO_V3_IF_V2_BACKGROUND_INHERITED`。V2 A2的身份是旧host/点缺陷/对象合同下的
`MODEL_DEPENDENT_RESIDUAL`，不是可直接移植的材料常数。

替代合同在读取6/48 h状态前，只用AQ七点对H-P0以及H-P2的[100]、[110]、[111]和
角平均代理分别重标定B0/B1/B2/B4/BU，每个候选最多一个参数。按逐host/逐AQ结构
`Delta AICc <= 2`保留103个非概率成员；所有保留拟合的AQ MAPE为
`3.657--4.146%`，但按`0.015 W m^-1 K^-1`分析不确定度的绝对卡方均为差拟合，故必须
保留`POOR_ABSOLUTE_CHI2_WARNING`，不能把相对AICc胜者称为唯一真实背景。H-P1仍因缺少
资格化branch-resolved PbTe host合同而不可用。

关闭全部动态resolved散射后，所有103个替代合同成员在573.15 K的上限为：6 h
`0.855524/0.885191/1.420703`、48 h
`0.855322/0.884579/1.419865 W m^-1 K^-1`（min/median/max）。14个成员越过6 h与
48 h必要端点门，因此宽诊断包络状态为
`PASS_V3_ABSOLUTE_ENDPOINT_NECESSARY_CONDITION`；但14个通过者全部依赖H-P2的303.2 K
张量温度外推和`SINGLE_APT_COUNT_FRACTION_DIAGNOSTIC` AQ结构案例，正式
`SOURCE_LITERAL_BOUND`通过数为0。H-P0的48 h最大上限仅为
`0.9820424678784576`。所以该PASS只表示替代合同在宽代理包络中未被数学上排除，绝不
是V3正式预测或实验复现。

对14个端点通过成员，6 h仅校准一个非负D0/D2/D4测试谱幅度，再精确Debye反解48 h所需
率比例，得到D0 `0.032046/0.162909/0.334153`、D2
`0.082232/0.303813/0.494168`、D4 `0.082763/0.285531/0.461176`
（min/median/max）。这些范围与PF的`Sv=0.395306`、mid-q=`0.306247`、
high-q=`0.458338`存在重叠，只构成结构衰减量级一致性；冻结V1 full-PSD无位错结果的
48 h/6 h比仅为`0.997541`，密度对比粗化自身不能给出实验的21.18%恢复。

现有来源只能构造Ag2Te的一维各向同性标量控制轨迹（`rho=8200 kg m^-3`、523 K单点
`B=18.9 GPa`和实验剪切模量趋势），不能给出高温密度范围、立方各向异性、branch
damping和界面透射的独立有来源多维包络。资格化单颗粒数值内核的27个控制中有21个满足
宽数值门，但物理材料权威通过数为0；`v3_feasible_parameter_region.csv`因此明确记录
`NO_QUALIFIED_PHYSICAL_PARAMETER_REGION`。材料结论为
`BLOCKED_V3_WIDE_SOURCED_MATERIAL_ENVELOPE_UNAVAILABLE_NUMERICAL_CONTROL_HAS_LEVERAGE`，
不得升级为`PASS_V3_MATERIAL_ENVELOPE_FEASIBILITY`。按本次任务的决策合同，在可识别材料
包络形成前应停止或缩减完整昂贵atomistic campaign，只执行独立于Sheskin 48 h的最小
高温`C11/C12/C44`、branch damping与OR材料识别门，再重新冻结和审计。

最终状态为
`PASS_V3_ABSOLUTE_ENDPOINT_NECESSARY_CONDITION_MATERIAL_AUTHORITY_BLOCKED`。所有关键输出
由同一冻结输入独立生成两次并要求逐字节一致，输入、脚本和输出SHA-256均登记；历史报告
未被覆盖。

### 13.37 400³ PSD/空间/密度阶梯的静态整数库存门（2026-08-03）

为拟议的400³、T380、有弹性、实验matrix锚定的21个唯一PSD/空间/密度
case，已在任何400³ fixture物化或Slurm提交之前，对冻结的15-entry
Method-1 quarter-nm profile库执行只读整数h-volume可行性审计。V5 pilot、
原始profile、历史fixture和checkpoint均未修改。

该新任务同时要求：

```text
target_h_volume_nm3=1531490.9025410344
relative_h_volume_error<=1e-10
```

但冻结库的每个注册半径满足`q=4R`为整数，其理想等效体积为
`(pi/48) q^3`。记录的profile真实h-volume只在该离散格点上加了正的、
约`1.9e-8`--`2.0e-7 nm³`的单profile修正。目标体积距离最近理想格点
约`2.3e-2 nm³`；即使给每个profile加上最有利的记录修正，也远大于
允许的`1.5314909e-4 nm³`绝对误差。

对N=256/512/640/704的严格下界分别为：

```text
0.0231318125, 0.0230807208, 0.0230551750, 0.0230424021 nm³
```

因此这在PSD形状、空间seed或优化器之前就是共同的数学不可行性；不得
通过缩放、插值或修改冻结profile/目标库存绕过。所有21个case没有物化，
双队列42个逻辑提交没有执行。当前状态为：

```text
BLOCKED_PROFILE_LIBRARY_RANGE
```

证据：

```text
scripts/audit_pf_400cube_psd_spatial_density_ladder_integer_feasibility_v1.py
reports/pf_400cube_psd_spatial_density_ladder_production_v1/
integer_feasibility_preflight_20260803/
```

这不影响仍在独立运行的400³ V5 pilot。若后续要重开阶梯campaign，必须由
用户明确选择并冻结一个兼容的目标h-volume/离散容差合同，或先构建新的
已验收profile半径库；二者都不能在本任务中静默替换。

### 13.38 workstation/cluster实时运行状态（2026-08-04）

本节是对13.37之后实时队列和运行目录的只读核对，检查时间为
`2026-08-04 03:34 CST`。它不把队列中的`RUNNING`或静态qualification
误写成最终生产PASS。

#### workstation (`workstation-tail`, 主机`fuxin`)

当前没有活动的`main_cuda`进程，也没有GPU compute进程。246³主生产目录
仍保留旧的未收尾状态：

```text
/home/zhiheng/tmp/pf_246cube_6h48h_B_workstation_v1_20260731
status=RUNNING_246CUBE_6H48H_PRODUCTION_V1

/home/zhiheng/tmp/pf_246cube_6h48h_C_exp_matrix_xAg0p0062_v1_20260731
status=RUNNING_246CUBE_6H48H_PRODUCTION_V1
```

上述两个`RUNNING`文件的最后更新时间仍为`2026-07-31`，实际输出只到
`segments/step_3633`，因此它们不能被解释为完成或生产PASS。B的后续
`commit1c08f9e`目录另外记录了：

```text
CANCELLED_BY_USER_PRESERVED_OUTPUT_V1
last_completed_checkpoint_step=10899
interrupted_segment_target_step=14532
```

workstation上已完成的辅助资格任务必须与主生产分开定级，包括
`pf_246cube_6h8h_A_v1`、`pf_246cube_short_restart_A_v1`、
`transport_v3_pf_descriptors_final_A/B_20260802`以及
`sheskin_mechanics_replay_qualification_v2_20260802`的PASS。它们不构成
246³ B/C的6--48 h生产完成证据。

#### cluster (`cluster-direct`/`uvip-cluster`, 主机`login1`)

400³ PSD/空间/密度阶梯的实际Slurm数组已经提交到`gpu_uvip`：

```text
array job=74232
element=74233 (74232_1)
state=RUNNING
node=gpu1
time_limit=24:00:00
```

在检查时，`74232_1`已运行约14小时，数组的`74232_[2-21%1]`仍为
`PENDING`。当前campaign目录为：

```text
/data/home/luozhiheng/tmp/pf_400cube_psd_spatial_density_ladder_production_v1_20260803
```

该目录的静态和短资格状态为：

```text
21/21  fixture static qualification PASS
9/21   short qualification PASS
0/21   final 6--48 h production PASS
```

当前运行case的状态文件为：

```text
attempts/001/gpu_uvip_74233_1/status.txt
RUNNING_400CUBE_PSD_LADDER_PRODUCTION_V1
```

因此cluster主生产尚未完成；必须等待数组任务结束并逐case通过最终
segment、restart、6--8 h和6--48 h分析门后，才能生成最终production
authority。此前`74006`数组失败、`73850`被取消等历史状态继续保留，不能
与当前`74232`运行混合统计。

截至本节记录，项目运行状态应概括为：

```text
WORKSTATION_246CUBE_MAIN_PRODUCTION=NOT_COMPLETE_NO_ACTIVE_PROCESS
CLUSTER_400CUBE_PSD_LADDER=RUNNING_ARRAY_PARTIAL
FINAL_6_TO_48H_PRODUCTION_AUTHORITY=NOT_AVAILABLE
```

### 13.39 当前工作分支身份（2026-08-05）

当前本地工作分支已切换为：

```text
CUDA_STO_PF-transport-v3-polarization-anisotropic
```

该分支由原有工作树创建，当前尚未关联`origin`远程跟踪分支。切换分支时
没有清理、回滚或覆盖已有改动；工作树中的未提交源码、脚本、数据和报告
仍属于同一项目工作集，后续提交或发布前必须按文件来源和provenance单独
核对。分支名称只记录版本身份，不改变当前研究边界、冻结参数或
`FINAL_6_TO_48H_PRODUCTION_AUTHORITY=NOT_AVAILABLE`的状态。

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

### 13.40 后续400³ PSD阶梯任务的bulk VTK输出政策（2026-08-05）

经用户重新确认，后续新提交的400³ PSD/空间/密度阶梯任务也继续保持
`CUDA_STO_SUPPRESS_VTK_OUTPUT=1`，不输出bulk VTK。checkpoint仍是权威场状态；
如需可视化，另行使用只读的checkpoint导出工具，并保留源checkpoint哈希和
provenance，不改变PF生产任务、checkpoint或输运合同。已经运行或已经提交的
任务（包括当前的74017、74964及74232数组）不修改、不重排、不追溯改变。

### 13.41 400³ PSD阶梯Case 001/002的V2端点热输运审计（2026-08-05）

对Case 001/002的完成checkpoint执行了条件性transport-V2端点审计。冻结
18个AQ背景/界面成员，保持`A_N=1.5`、`S11=S13=0`，未使用Yu
`0.1172768`缩放。使用完整端点PSD和等效球几何`Sv`计算了V2的`M0`与`MI`；
由于当前checkpoint没有同一accepted field时间层的mechanics replay，`MS/MIS`
不计算，也不以checkpoint内对应`n-1`的warm displacement替代。

在573.15 K、PF时变matrix条件下：

```text
case 001: M0 0.873537 -> 0.871782, Δκ=-0.001755 W m^-1 K^-1
          MI 0.817802 -> 0.851336, Δκ=+0.033534 W m^-1 K^-1
case 002: M0 0.873538 -> 0.871955, Δκ=-0.001583 W m^-1 K^-1
          MI 0.817802 -> 0.852211, Δκ=+0.034409 W m^-1 K^-1
```

这说明在当前001/002端点中，full-PSD/点缺陷通道本身给出轻微负变化；V2
界面项因`Sv`从约`7.98e6`降到`2.63e6/2.54e6 m^-1`而给出约4.1--4.2%
的正变化。该结果状态为
`CONDITIONAL_V2_M0_MI_ENDPOINT_AUDIT_V1`，不是生产authority，也不声称
绝对实验热导率复现。结果和哈希保存在
`reports/pf_400cube_v2_case_pair_audit_v1/`；端点粒子身份输入仍标记为
endpoint-only blocked，待可流式完成全checkpoint链审计后再决定是否升级。

### 13.42 THERMODYNAMIC_CONTRACT_FREEZE_V1跨环境审计（2026-08-05）

已按用户指定的严格跨环境合同流程完成Stage A只读审计，覆盖本地工作树、
GitHub远端、workstation和cluster，并保留完整审计包：

```text
thermodynamic_contract_freeze_v1/
```

审计确认当前生产路径仍使用legacy正则溶液参数：

```text
L(T)=41212.9-18.05*T J/mol
T=653.15 K: xB_eq=0.004664951821454188
T=653.15 K: xAg_eq=0.004654096254055399
```

用户请求中给出的`ΔH≈41504.291 J/mol`、`ΔS≈18.469277 J/(mol K)`和
`xB_eq≈0.0046492610`未在本地、Git对象、远端引用、workstation或cluster
中找到可核验的完整原始拟合源、四点数据、残差/协方差和版本绑定证据；因此
不能将其提升为出版权威，也不能把legacy值静默改写为exact-v1。17条legacy
字面量路径和跨环境源码/二进制不一致均已登记，尚未解析。

本次没有修改生产代码、配置、二进制、checkpoint、运行队列、远端文件，也
没有提交、打tag或推送。当前400³生产合同门状态为：

```text
FINAL_STATUS=BLOCKED_AUTHORITATIVE_CONTRACT_CONFLICT
READY_FOR_400_PRODUCTION=NO
CONTRACT_HASH=NONE_BLOCKED
WORKSTATION_DEPLOYED=NO
CLUSTER_DEPLOYED=NO
```

在获得可复核的exact-fit源证据、统一源码/参数/二进制身份并通过敏感性与
跨环境测试前，所有既有结果只能按legacy或mixed/unknown provenance使用，
不得宣称exact-v1出版复现。解阻塞所需的完整证据和后续patch顺序已写入
`thermodynamic_contract_freeze_v1/02_authoritative_contract_decision.md`、
`10_local_patch_plan.md`和`19_final_acceptance_report.md`。

### 13.43 外部Solubility Calibration exact-fit证据补充（2026-08-05）

本节补充并 supersede 13.42 中“本地项目/远端环境未发现 exact source”的
范围限定：外部校准路径现在已被纳入证据边界，但不改变13.42的跨环境部署
阻塞结论。

用户随后指定外部校准路径：

```text
/Users/heng/Desktop/1_Solubility_Calibration
```

该路径包含四点原始数据、exact拟合脚本、最终参数CSV和终端输出。对原始
四点独立复算得到：

```text
Delta_H=41504.29119633958 J/mol
Delta_S=18.469276826409214 J/(mol K)
L(653.15 K)=29441.083037170407 J/mol
xB_eq(653.15 K)=0.004649261005504821
xAg_total_eq(653.15 K)=0.46384782574611727 at.%
```

这使exact值从“提示词候选”升级为`AUTHORITATIVE_EXACT_FIT_OUTPUT
candidate`，但尚未成为跨环境production/publication contract。校准工作区
自己的numerical conflict audit仍将以下项目列为阻塞：pointwise residual
CSV、完整文献/数字化不确定度、exact-vs-legacy对既有PF结果的matched
sensitivity，以及统一源码/参数/二进制/contract hash。故当前状态不变：

```text
AUTHORITATIVE_CONTRACT=CONFLICTED_EXACT_CANDIDATE_VS_LEGACY_RUNTIME
FINAL_STATUS=BLOCKED_AUTHORITATIVE_CONTRACT_CONFLICT
READY_FOR_400_PRODUCTION=NO
```

所有既有246³/400³结果继续按legacy或mixed/unknown provenance处理，不得
因为发现external exact fit而改写旧结果metadata或静默切换runtime。

### 13.44 用户授权的本地exact热力学参数补丁（2026-08-05）

用户明确要求将热力学参数直接改为最新校准值。当前分支本地源文件已更新为：

```text
Delta_H=41504.29119633958 J/mol
Delta_S=18.469276826409214 J/(mol K)
L(T)=Delta_H-Delta_S*T
T=653.15 K: xB_eq=0.004649261005504821
```

`thermo_utils.h`新增本地候选常量和启动日志标识
`exact_candidate_v1_local_unfrozen`；`main_cuda.cu`、`Unit_Psedobinary.py`及
相关分析脚本同步切换。Python语法检查和独立根复算通过，但本机无`nvcc`，
因此没有生成新CUDA binary。workstation/cluster、旧binary、checkpoint、旧
结果和运行队列均未修改；跨环境合同仍为：

```text
LOCAL_PARAMETER_PATCH=APPLIED
CONTRACT_HASH=NONE_BLOCKED
REMOTE_DEPLOYMENT=NO
READY_FOR_400_PRODUCTION=NO
```

本地源补丁不改变旧结果的legacy/mixed provenance，也不等同于完成发布合同
冻结；部署前仍需构建隔离binary并通过跨环境identity、restart和敏感性测试。

### 13.45 21-case期间维持legacy热力学合同（2026-08-05）

用户最终取消workstation/cluster的exact-candidate部署，以避免分段运行的
21-case campaign在共享binary切换时混入两套合同。独立cluster构建任务已取消，
两个远端临时exact工作树已移除；生产源码、binary、运行任务和排队任务未改动。

当前操作边界为：

```text
LOCAL_GIT_BRANCH=exact_candidate_source_patch
WORKSTATION_RUNTIME=legacy_41212p9_minus_18p05T
CLUSTER_21_CASE_RUNTIME=legacy_41212p9_minus_18p05T
REMOTE_EXACT_DEPLOYMENT=NO
IN_FLIGHT_RESULT_PROVENANCE=LEGACY
```

本地分支可以发布exact-candidate源码用于后续审查，但不构成production freeze，
也不得改写21-case或历史结果标签。exact切换只能在当前campaign结束后的新任务
边界进行，并重新绑定source、parameter、binary、fixture、checkpoint和analysis
provenance。
