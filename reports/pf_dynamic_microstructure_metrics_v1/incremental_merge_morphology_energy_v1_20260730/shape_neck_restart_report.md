# 逐颗粒形貌张量、合并颈部持续性与弹性能重启审计

## 总体结论

`PASS_REQUESTED_AUDITS_WITH_16H_DIFFUSE_NECK_CLASSIFICATION`

三个请求均已完成：

- 6–8 h、30 帧、2463 条逐颗粒形貌张量记录已生成；
- 15–16 h 身份事件完成十个 \(h(\phi)\) 阈值的颈部持续性检查；
- 246³ 弹性开启的 512 步连续/重启短对照通过。

最重要的科学修正是：16 h 的 2→1 连通仅在 \(h\le10^{-3}\) 成立，属于 diffuse-neck onset；到 17 h 才形成可在 \(h=0.05\) 下保持的较强颈部。因此，16 h 不应写成“实体颗粒已经完成并合”。

## 1. 6–8 h 逐颗粒惯性张量和取向

状态：`PASS_PARTICLE_SHAPE_TENSOR_6_8H_V1`

### 定义

每个颗粒使用周期展开坐标和 \(h(\phi)\) 体积权重。输出同时包括：

- 形貌协方差张量 \(C_{ij}=\langle r_i r_j\rangle_h\)，单位 nm²；
- 几何惯性张量
  \[
  I_{ij}=\int h(\phi)(r^2\delta_{ij}-r_i r_j)\,dV,
  \]
  单位 nm⁵；
- 三个主特征值和完整正交主轴；
- 等效椭球半轴 \(a_i=\sqrt{5\lambda_i}\)；
- \(a/c\)、\(a/b\)、\(b/c\) 和 triaxiality；
- 主轴对模拟盒 `[100]`、`[010]`、`[001]` 的夹角。

主轴没有正负方向，CSV 中的符号只用于确定性序列化。当

\[
\frac{\lambda_1-\lambda_2}{\lambda_1}<0.05
\]

时，颗粒接近球形或轴简并，取向被标记为 ill-conditioned，不作物理解读。

### 结果

| 时间 | 颗粒数 | 取向可辨颗粒 | 平均 \(a/c\) | 平均等效半轴 \(a,b,c\) (nm) | 可辨主轴 |
|---:|---:|---:|---:|---|---|
| 6.000 h | 96 | 0 | 1.00000 | 9.745, 9.745, 9.745 | 近球形，未定义 |
| 6.141 h | 96 | 96 | 1.21220 | 10.326, 9.885, 8.534 | 全部最近 `[100]` |
| 6.987 h | 86 | 84 | 1.38577 | 11.083, 9.991, 7.908 | 全部最近 `[100]` |
| 8.000 h | 59 | 58 | 1.44639 | 12.916, 11.392, 8.836 | 全部最近 `[100]` |

完整 2463 条记录中：

- \(a/c\) 范围：1.0000002–1.5468288；
- 平均 \(a/c\)：1.35265；
- 93.666% 的记录满足主轴可辨条件；
- 所有 2307 条可辨记录的最近盒轴均为 `[100]`；
- 8 h 时对 `[100]` 的平均偏角为 3.856°，P90 为 4.945°。

这表明初始球形颗粒在弹性耦合作用下很快选择 x/[100] 方向并拉长。该结论依赖“模拟盒轴等同晶体轴”的当前合同，可以作为模型内取向选择证据，但不能直接等同于实验织构或多取向变体分布。

## 2. 15–16 h 多阈值颈部持续性

状态：`TREND_ONLY_DIFFUSE_TAIL_NECK`

检查阈值：

```text
1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.25, 0.5, 0.75
```

使用 parent 46 和 68 在 15 h 的质心附近局部 \(h\) 最大值作为两个锚点，并对周期六邻域连通分量逐阈值检查。

| 时间 | 最大连通阈值 | 直线最低 \(h\) | 判定 |
|---:|---:|---:|---|
| 14 h | 无 | \(1.39\times10^{-6}\) | 两颗粒分离 |
| 15 h | 无 | \(2.60\times10^{-5}\) | 两颗粒分离 |
| 16 h | 0.001 | \(2.19\times10^{-4}\) | diffuse neck 已出现 |
| 17 h | 0.05 | \(1.16\times10^{-2}\) | 颈部明显增强 |

16 h 与 17 h 共同持续的阈值只有 `1e-4, 5e-4, 1e-3`。在预注册的物质颈部门槛 \(h\ge0.005\) 下，16 h 尚未连通；17 h 已连接至 \(h=0.05\)。

因此正确叙述是：

> 颗粒 46 与 68 在 15–16 h 之间进入低 \(h\) diffuse-neck 接管，随后在 16–17 h 之间形成更强的物质颈部。16 h 的 tracking-threshold 2→1 lineage 不等同于已完成实体并合。

## 3. 弹性能诊断连续/重启对照

状态：`PASS_PF_ELASTIC_DIAGNOSTIC_AND_FIELD_RESTART_V1`

Cluster job `72974`：

- 连续：0→512 步；
- 分段：0→256 步，checkpoint，再从 256→512 步；
- 网格：246³；
- 弹性：开启；
- `dt_code=0.02`；
- 诊断间隔：64 步；
- 总物理时间：507.354 s；
- 连续运行速度：0.440999 s/步。

八个注册步 `64,128,...,512` 上，以下字段连续轨迹与“前半段+重启段”逐字符串一致：

- `time`；
- `mean_elastic_energy`；
- `max_elastic_energy`；
- `stress_hydro_min/max`；
- `mean_xBtot_end_step`；
- `total_delta_mass_step`。

最终场与 checkpoint：

| 对象 | SHA-256 | 连续/重启 |
|---|---|---|
| `phi_512.vtk` | `5edf289012ef4063d77847f27c4204a270774bbffcb76dd863ae1e7d3052c9af` | bytewise PASS |
| `xB_512.vtk` | `8fdf4ea7485cfad26240cf9ac940b823741b9cc61daefb866911ba5f789d7588` | bytewise PASS |
| `xBtot_512.vtk` | `17a29fd880fce524eff64b72dbe9a24f8099b0df56e6f9a857dbbe0b6df8a36c` | bytewise PASS |
| final checkpoint | `ec22d8046fcea72a217ba1218bc552111f6f3cb8670f4db5476ddaeac600ed00` | bytewise PASS |

这证明新增弹性能诊断既不会改变场演化，也能在 checkpoint/restart 后无缝恢复同一条诊断轨迹。

## 最终标记

```text
particle_shape_tensor_status=PASS_PARTICLE_SHAPE_TENSOR_6_8H_V1
shape_snapshot_count=30
shape_particle_row_count=2463
orientation_well_conditioned_fraction=0.9366626066
conditioned_closest_axis=[100]

neck_persistence_status=TREND_ONLY_DIFFUSE_TAIL_NECK
max_connected_threshold_15h=0
max_connected_threshold_16h=0.001
max_connected_threshold_17h=0.05
completed_material_merge_at_16h=false

elastic_restart_status=PASS_PF_ELASTIC_DIAGNOSTIC_AND_FIELD_RESTART_V1
elastic_diagnostic_string_exact=true
final_fields_bytewise_equal=true
checkpoint_bytewise_equal=true

full_48h_rerun=false
analysis_job=72975
restart_job=72974
final_status=PASS_REQUESTED_AUDITS_WITH_16H_DIFFUSE_NECK_CLASSIFICATION
```
