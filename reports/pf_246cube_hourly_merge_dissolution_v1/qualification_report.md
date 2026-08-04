# A 场每小时检查点 merge-aware / 溶解资格审计

## 结论

```text
final_status=PASS_246CUBE_HOURLY_MERGE_DISSOLUTION_AUDIT_V1
A_field_rerun=false
exact_event_time_claimed=false
```

本审计只读取 A 场已有的 6--48 h 初态与 44 个每小时检查点；没有重跑
PF、没有修改物理参数、没有覆盖原始输出。Cluster CPU 独立复核作业
`73425` 于 1 s 内完成，退出码为0，stderr为空。Cluster输出与两次本地重复
运行的全部审计文件逐字节一致。

## 为什么旧审计会阻塞

旧C++追踪器要求粒子在消失前至少留下三次连续递减的体积记录。每小时输出
下，很多小粒子在相邻两个快照之间就从连通域中消失，因此旧规则把91个
中阈值消失中的61个标为“时间证据不足”。这并不等于它们重新出现、发生
质量丢失或产生新颗粒。

第一次严格组合审计还正确暴露了旧追踪器的另一个表达缺陷：合并组的
`stable_id`在阈值颈部重新打开后被复制给两个子连通域，造成同一时刻重复
stable ID。该尝试被保留在`initial_rejected_attempt/`，没有被包装成PASS。

## 冻结的审计合同

审计使用同一组场的三个嵌套连通阈值：

```text
low support:  h > 1e-4
resolved core: h > 1e-3
strong core:   h > 5e-3
```

- 溶解只登记为相邻快照之间的有界区间，不推断小时内的精确事件时刻；
- `h>1e-3`身份消失必须有更强`h>5e-3`不晚于它消失的证据；
- `h>1e-4`只用于检查弥散尾部，不单独定义resolved粒子核心；
- merge必须通过父体重叠比例和子体覆盖比例门；
- merge后阈值颈部重新打开时，仅在周期质心到最后独立身份锚点存在唯一
  一一映射、且没有跨越Voronoi门时，才恢复原身份；
- 极弱交叉边只有在与同一步伪split成对、且不满足主导重叠门时才可剪除；
- 任何无法唯一恢复的split、new component、身份重现、事件账本缺失、阈值
  顺序倒置、质量越界或字段非有限都立即fail closed；
- 身份不按体积排序。

## A 场结果

```text
snapshot_count=45
initial_identity_count=96
final_resolved_core_component_count=5
final_resolved_core_identity_count=5
resolved_core_dissolution_count=91
maximum_dissolution_interval_s=3600.034504381788
threshold_contact_group_count=1
persistent_strong_core_merge_group_count=0
unresolved_split_count=0
new_component_count=0
identity_reappearance_count=0
max_mass_relative_error=3.272655446666275e-13
```

91个溶解身份的证据分类为：

- 30个：旧时间递减规则和三阈值完全消失同时通过；
- 60个：三阈值在同一小时区间内完全消失，但旧规则因小时采样过疏没有
  三个先验体积点；
- 1个（`P078`）：strong/medium核心在25--26 h区间消失，但`h>1e-4`
  的低幅尾部仍与`P087`相连。

`P078/P087`的拓扑历史为：低阈值约16 h接触，中阈值约19 h接触，强阈值
约22 h接触；强阈值约24 h重新分开，中阈值约25 h重新分开，随后`P078`
核心在25--26 h区间消失。周期质心恢复身份的最大位移为3.0324 nm，最小
赋值优势为76.5572 nm。强阈值25 h处另有一条仅6体素的交叉边，其父/子
占比分别低至`5.305e-4`和`8.257e-5`，被严格登记为弱重叠边，而非物理
merge/split。

因此该组只能登记为：

```text
THRESHOLD_CONTACT_NOT_PHYSICAL_MERGE
```

不能把它计为真实颗粒合并。48 h低阈值仍有5个support连通域、承载6个
历史身份；中/强阈值均为5个resolved core身份。这一差异由`P078`附着的
低幅尾部解释。

## 验证矩阵

- 合成正例：旧tracker为BLOCKED、小时区间溶解仍可严格PASS；
- 9个负例全部fail closed：真实split、新component、未资格merge、缺失
  dissolution事件、身份重现、重复身份、阈值顺序倒置、质量漂移、缺失
  checkpoint；
- 历史真实C 6--8 h三阈值案例回归PASS；
- A场本地双运行逐文件SHA-256一致；
- Cluster作业73425与本地输出逐文件SHA-256一致。

## 状态边界

本报告验收的是A场的每小时谱系、merge/contact分类和溶解区间。它不改写
A原始运行根中的旧`BLOCKED`文件，也不自行宣布整个A生产链已经通过；完整
生产PASS仍需让生产汇总器显式接纳本审计合同，并重新组装质量、零模、
checkpoint、能量和科学观察量的总判定。

