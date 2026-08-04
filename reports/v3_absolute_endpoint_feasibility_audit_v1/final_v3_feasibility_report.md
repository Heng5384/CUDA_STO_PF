# V3绝对端点可达性与背景可转移性最终报告

## 结论

本审计得到三个必须分开的结论：

1. **数学NO-GO（继承V2背景）**：`NO_GO_V3_IF_V2_BACKGROUND_INHERITED`。冻结V2 M0在573.15 K的48 h上限为`0.871682`，低于实验`1.03`；任何新增非负散射都不能把它向上推。
2. **必要条件PASS（替代合同）**：`PASS_V3_ABSOLUTE_ENDPOINT_NECESSARY_CONDITION`。重新定义host/background并只用AQ重标定后，`14/103`个非概率成员的无动态48 h上限达到1.03。但通过者全部属于H-P2温度外推代理，并且全部依赖`SINGLE_APT_COUNT_FRACTION_DIAGNOSTIC` AQ结构案例；`SOURCE_LITERAL_BOUND`通过数为`0`。这只是宽诊断包络中的数学可达性，不能称为正式V3预测。
3. **物理材料可行性尚未PASS**：`BLOCKED_V3_WIDE_SOURCED_MATERIAL_ENVELOPE_UNAVAILABLE_NUMERICAL_CONTROL_HAS_LEVERAGE`。V3单颗粒内核的一维标量控制轨迹有`21`个数值门通过点，但这不是题目要求的宽、多维、有来源材料包络；高温Ag2Te单晶张量、branch damping和界面OR仍无权威包，因此不存在可登记的“内部物理可行区域”，不得写成`PASS_V3_MATERIAL_ENVELOPE_FEASIBILITY`。

## 核心数值

- V1冻结复现：`1.2980916621 -> 1.2948991542 W m^-1 K^-1`。
- V2 M0冻结复现：`0.8733452622 -> 0.8716818275 W m^-1 K^-1`。
- V2 MIS冻结复现：`0.8208832451 -> 0.8490371923 W m^-1 K^-1`。
- 继承V2背景最大恢复：`0.021682 W m^-1 K^-1`，即`2.551%`，远低于实验`+21.18%`。
- V3替代合同无动态上限（所有保留成员）6 h min/median/max：`0.855524/0.885191/1.420703`。
- V3替代合同无动态上限（所有保留成员）48 h min/median/max：`0.855322/0.884579/1.419865`。

## 背景身份

V2 A2是在Yu单平均声速host、旧点缺陷合同和density-only AQ对象包络下反演的模型残差，不是可转移材料常数。原样继承只保留为负对照。V3候选B0/B1/B2/B4/BU均在各自host下只用AQ七点重新识别；冻结manifest写出后才加载6/48 h权威状态。所有散射率保持非负，未使用Yu `0.1172768/0.119`位错比例，也未调整`A_N=1.5`。

## 科学边界与决策

V3 cannot solve the amplitude problem by being added on top of the frozen V2 background, because the 48 h no-dynamic-scattering upper bound is already below experiment.

Rebuilding the host/background contract removes the mathematical hard cap, but does not prove that physically realistic Ag2Te parameters will produce the required recovery.

按本任务的决策规则，当前证据不足以建议直接继续完整、昂贵的高温Ag2Te DFT/AIMD/TDEP campaign。应先把工作缩减为不读取Sheskin 48 h的最小材料识别门：取得正定的`C11/C12/C44(T)`、branch/direction速度、阻尼及界面OR稳定性；只有这些量能构成有来源的多维包络并通过Born/T-matrix物理门，才重新冻结V3、重复本审计并决定是否进入完整盲预测。不得声称绝对实验晶格热导率复现。

最终状态：`PASS_V3_ABSOLUTE_ENDPOINT_NECESSARY_CONDITION_MATERIAL_AUTHORITY_BLOCKED`。
