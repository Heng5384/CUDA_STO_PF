# B 场每小时检查点 merge-aware / 溶解资格审计

## 结论

```text
final_status=PASS_246CUBE_HOURLY_MERGE_DISSOLUTION_AUDIT_V1
B_field_rerun=false
exact_event_time_claimed=false
```

本审计使用与A完全相同的审计脚本、`h>1e-4/1e-3/5e-3`三阈值合同和
每小时事件区间语义，只读取B场已有初态及44个checkpoint。Cluster CPU作业
`73426`运行3 min 36 s，退出码0，stderr为空。重新生成的低阈值五个原始
追踪文件与B生产目录中已有版本逐文件SHA-256一致，证明没有改写B场。

## 结果

```text
snapshot_count=45
initial_identity_count=96
qualified_dissolution_count=87
final_low_support_component_count=8
final_strong_identity_lower_bound=8
final_medium_identity_upper_bound=9
threshold_contact_group_count=2
persistent_strong_core_merge_group_count=0
unresolved_split_count=0
new_component_count=0
max_mass_relative_error=2.8581972897407958e-13
```

87个严格资格溶解中：

- 21个同时通过旧连续体积递减证据；
- 66个由三阈值的有序小时区间消失闭合，但旧规则因采样稀疏未资格；
- 没有身份重现、事件账本缺失或质量越界。

## 两组接触

### P077/P086

只在`h>1e-4`低幅support上于约26 h接触，约32 h重新分开；周期质心恢复
为唯一一一映射。中、强阈值没有形成共同核心，因此严格分类为低幅尾部
接触，不是物理merge。

### P078/P079

低/中/强阈值分别约在14/16/19 h连通，强阈值约20 h重新分开；到48 h，
低/中阈值仍保留两个历史身份的共同support，而强阈值只保留`P079`核心。
因此该组同样登记为：

```text
THRESHOLD_CONTACT_NOT_PHYSICAL_MERGE
```

`P078`尚不能按当前每小时证据严格登记为完全溶解，也不能计作独立强核心。
所以B的48 h结果必须报告为8个连通域、8个确定强核心身份、最多9个中阈值
历史身份，不能武断写成9个独立resolved粒子。

## 验证

- B本地双运行全部审计文件SHA-256一致；
- cluster作业73426独立PASS；
- 低阈值重建与原B生产后处理逐文件一致；
- 与A使用同一审计器SHA-256：
  `e86bfea8a9088f4980b33b98f32239783a81adbc9b2cc81c946ec51d9e2317f9`。

