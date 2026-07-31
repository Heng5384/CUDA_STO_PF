# Stage 1 — Yu et al. 2024 全文精读与项目对标矩阵

## 文献
Yuan Yu et al., “Ostwald Ripening of Ag2Te Precipitates in Thermoelectric PbTe: Effects of Crystallography, Dislocations, and Interatomic Bonding”, Advanced Energy Materials 14 (2024) 2304442. DOI: 10.1002/aenm.202304442.

## 样品历史
- 名义组成：(PbTe)0.97(Ag2Te)0.03。
- 熔炼：12 h 升至 1273 K，保温 6 h；缓冷至 973 K，均匀化 48 h；冰水淬火。
- 粉末热压：923 K，总时长约 30 min，其中 45 MPa 约 15 min，Ar–7%H2；随后冰水淬火。该状态定义为 AQ。
- 退火：653 K（380 °C）48 h，120 torr Ar–7%H2 密封石英管；随后冰水淬火。该状态定义为 annealed。
- 输运测试：300–575 K，低于退火温度以减少测试时组织变化。

## 组织与成分
| 指标 | AQ | Annealed |
|---|---:|---:|
| 小型 Ag-rich 对象 | 球形，1–5 nm | 大幅减少 |
| 小对象数密度 | 7.5e24 m^-3 | 2.5e19 m^-3 |
| 小对象组成 | Ag-rich core，最高约30 at.% Ag；Pb:Te约1:1 | 主文未给完整分布 |
| 大析出物 | plate-like，平均半径20–30 nm | lath-shaped；代表对象约500×100×50 nm |
| 大析出物数密度 | 1.9e21 m^-3 | 9.2e18 m^-3 |
| matrix Ag | 0.69±0.11 at.% | 0.34±0.05 at.% |
| 位错密度 | 未观察到 | 3.5e11 cm^-2 |

小对象下降约3e5倍，大析出物下降约2.1e2倍。退火后大析出物接近化学计量 Ag2Te，且出现 Ag 装饰位错/Cottrell atmospheres。

## 取向与界面
- OR-a：[201]Ag2Te ∥ [001]PbTe；(200)Ag2Te/(002)PbTe；失配 +4.6%。
- OR-b：[010]Ag2Te ∥ [̅110]PbTe；(020)Ag2Te/(220)PbTe；失配 −2.2%。
- OR-c：[001]Ag2Te ∥ [110]PbTe；(204)Ag2Te/(220)PbTe；失配 −1.7%。
- 退火 lath 最长方向对应 OR-c，次长对应 OR-b，最短对应 OR-a。
- DFT 扫描72个界面平移构型，最低界面能约168–169 mJ m^-2。

注意：显微与DFT讨论的是淬火后室温单斜 P21/c Ag2Te；380–400 °C 时效期间的高温 Ag2Te 相、弹性常数和本征应变必须独立审计。

## Debye–Callaway
kappa_lat 使用 Debye–Callaway 积分；总散射率为：
1/tau_tot = 1/tau_U + 1/tau_GB + 1/tau_PD + 1/tau_Pre + 1/tau_Dis。

AQ 谱学分解使用：Umklapp、晶界、点缺陷、小型对象、大型 plate-like Ag2Te。论文认为小对象和大 plate 对 kappa_lat 降低贡献超过一半。

Annealed 谱学分解使用：Umklapp、晶界、点缺陷、析出物、位错。析出物散射变弱，Ag 装饰位错的应变场散射成为主要缺陷贡献。

详细散射公式和参数位于 Supporting Information；当前主文不足以逐参数复现。

## 与本项目的差异化
Yu 2024 已经完成：AQ/48 h 实验粗化、两类对象数密度、形貌和OR、DFT界面、matrix Ag、位错密度、两状态Debye–Callaway分解、电子和热电性能关联。

本项目仍可成立的内容仅应是：
1. 三维守恒正向群体演化；
2. 受总库存约束的条件初始化；
3. 动力学可达 PSD、界面面积和空间相关性轨迹；
4. 初始群体不确定性传播与“记忆/遗忘”分类；
5. 比较 Nv、平均尺寸、Sv、PSD矩和完整PSD，寻找预测 kappa_L 的最低充分组织描述符；
6. 从组织快照解释升级为条件性 kappa_L(t) 轨迹预测。

## Claim boundary
- 当前模型是 resolved Ag2Te 群体的 coherent/near-coherent、bulk-diffusion-controlled baseline。
- 不显式模拟1–5 nm小型Ag-rich对象、位错管道扩散、位错形成或大颗粒共格性丧失。
- 不声称首次研究Ag2Te粗化、OR或Debye–Callaway析出散射。
- Yu 2024 的 AQ→48 h 数据不能与 Sheskin 2018 的6 h→48 h基体平台数据静默拼接。
- 在取得Supporting Information前，不启动对Yu 2024 Debye–Callaway模型的定量复现。
