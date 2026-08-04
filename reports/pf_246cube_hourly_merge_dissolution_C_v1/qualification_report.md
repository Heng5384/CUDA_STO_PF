# C场每小时merge-aware / 溶解资格审计

## 结论

```text
final_status=PASS_246CUBE_HOURLY_MERGE_DISSOLUTION_AUDIT_V1
C_field_rerun=false
exact_event_time_claimed=false
```

C场相场计算已经完整生成6 h初态与44个每小时checkpoint；原生产driver的
`BLOCKED_246CUBE_6H48H_PRODUCTION_DRIVER_V1`来自旧低阈值tracker无法处理
两个持续连通组，并非PF场、质量或checkpoint失败。本审计不重跑场，只用与
A/B相同的`h>1e-4/1e-3/5e-3`三阈值合同重新解释已有输出。

44个场segment的`status.txt`均为`PASS_246CUBE_6H48H_SEGMENT_V1`，全部
`stderr.log`为空且checkpoint哈希存在。因此“相场运行完成”和“旧driver
后处理返回2”必须分开解释，不能把旧driver退出码当作场模拟失败。

Cluster CPU作业`73448`运行3 min 37 s，退出码0、stderr为空。重建的低阈值
五个原始tracker文件与C生产目录逐文件SHA-256一致；本地双审计的所有文件
逐字节一致，并与cluster的`audit.json`哈希相同。

## 谱系结果

```text
snapshot_count=45
initial_identity_count=96
qualified_dissolution_count=91
final_low_support_component_count=3
final_surviving_historical_identity_count=5
threshold_contact_group_count=0
persistent_strong_core_merge_group_count=2
unresolved_split_count=0
new_component_count=0
max_mass_relative_error=3.6428004673166977e-13
```

两组事件都在低、中、强阈值依次连通，之后一直没有重新分开，且父颗粒到
共同子component的周期重叠充分，因此登记为资格通过的持续强核心merge：

- `P075/P081`：低阈值约7--8 h连通，中/强阈值约8--9 h闭合；
- `P090/P092`：低阈值约24--25 h连通，中阈值约25--26 h、强阈值约
  26--27 h闭合。

所以48 h必须同时报告两个数字：3个物理连通析出体，以及其中承载的5个
历史幸存身份。不能把5写成5个互相独立颗粒，也不能把两次merge误写为
额外溶解。其余91个初始身份均有三阈值有序消失区间；无身份重现、未解析
split、新component或账本缺失。

## C场主要观察量

| 时效 (h) | 连通体数 | 平均等效半径 (nm) | β体积分数 | `Sv` (nm^-1) | `M6` (nm^3) | 远场Ag (at.%) |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 96 | 9.54161 | 0.02392961 | 0.00742617 | 5.35782 | 0.620292 |
| 12 | 26 | 14.1924 | 0.02325497 | 0.00459794 | 21.6514 | 0.688348 |
| 18 | 17 | 15.7431 | 0.02339397 | 0.00386355 | 41.6171 | 0.674632 |
| 24 | 10 | 19.1945 | 0.02364556 | 0.00331032 | 66.4081 | 0.648857 |
| 36 | 5 | 23.9215 | 0.02380185 | 0.00260423 | 147.275 | 0.633062 |
| 48 | 3 | 29.8101 | 0.02391576 | 0.00230167 | 194.638 | 0.621465 |

C场表现出强烈粗化：连通体数下降96.875%，平均半径增加212.42%，`Sv`
下降69.01%，`M6`增加约35.33倍；β体积分数最终只比6 h低0.0579%，远场
Ag回到0.6215 at.%。因此它满足“近恒定析出相库存下的溶解/长大/粗化”
方向证据，但包含两次真实merge，不能单独称为纯LSW Ostwald路径。

12 h远场Ag达到0.6883 at.%，高于48 h实验带；目前没有12 h实验锚点，且
生产合同明确不因中间浓度越界停止。该峰值应保留为6 h handoff初态的早期
组分瞬态，不能通过案例专用参数调整消除。
