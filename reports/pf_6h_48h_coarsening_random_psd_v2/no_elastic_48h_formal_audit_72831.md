# T380 无弹性 6–48 h 正式趋势审计（作业 72831）

## 1. 审计结论

本报告按照用户授权，将作业 72831 登记为 **正式的无弹性 6–48 h
后成核粗化趋势审计**。该运行完成了注册的 12–48 h 续跑，并通过了
零模守恒和最终 checkpoint 写入检查。

```text
formal_no_elastic_48h_audit=PASS_USER_AUTHORIZED
run_completion=PASS
elasticity=OFF
GP_birth_release_paths=OFF
zero_mode_final_audit=PASS
mass_ledger=PASS
coarsening_trend=PASS_OBSERVED
```

需要单独保留以下来源标记：该作业的可执行文件哈希为
`e5b6adcb489c821529e1a606b9bcff624e6b30565f72c07bce1ba8c30c23c1a3`，
不是当前冻结源提交 `6b69895af2d1b86b99c57c5479ff767349c61efe` 对应的
正式二进制 `695e5bdbd418530648673aba1a141e2dc6727ee6b6d865b5749571acaca34f07`。
因此本报告是正式趋势审计证据，但不是当前冻结二进制的同源复验。

```text
source_matched_to_current_frozen_binary=NO
provenance_note=LEGACY_BINARY_HASH_RETAINED_EXPLICITLY
```

## 2. 运行配置

| 项目 | 值 |
|---|---|
| 作业 | 72831 |
| 分区 | `gpu_uvip` |
| 网格 | `246^3` |
| 物理网格间距 | `1 nm` |
| 界面宽度 | `lambda_sm=4 nm` |
| 温度 | `380 degC` |
| `dt_code` | `0.02` |
| `t0_diff` | `49.5463048 s` |
| 物理步长 | `0.990926096 s` |
| 弹性 | 关闭 |
| GP Birth/release | 关闭 |
| 总步数 | `152600` |
| 12–48 h续跑步数 | `130800` |
| 续跑墙钟时间 | `9025.827 s = 2 h 30 min 28 s` |
| 平均续跑速度 | `0.0690048 s/step` |
| 参数文件 SHA-256 | `e00a17194fd08c8a3d8db286272667c009a9ef908a951c2a9eb1e820f6236178` |

12–48 h续跑的物理时长为约 `129613.13 s = 36.0036 h`。结合6 h
初始接管状态，终点对应48 h实验年龄。

## 3. 守恒与零模审计

运行日志中的最终审计为：

```text
PF_ZERO_MODE_FINAL_AUDIT status=PASS
final_step=152600
target_mass_code=4.46608079999958049e+05
final_mass_code=4.46608079999958049e+05
mean_mass_error=0.00000000000000000e+00
last_lambda=4.43943534160448384e-10
pf_zero_mode_status=PASS_PF_CONSERVED_Y_ZERO_MODE_V1
```

日志中的质量摘要为：

```text
rel_drift=-2.312965e-16
primary=phi_update_splitting
```

由 VTK 快照重新读取得到的最大绝对质量偏差为
`7.031971040599716e-07`；该值是快照重建/读取诊断，不替代上面的
canonical zero-mode ledger。两者均保留，不进行掩盖或互换。

## 4. 6–48 h 微观结构趋势

| 指标 | 6 h | 48 h | 变化 |
|---|---:|---:|---:|
| resolved 粒子数 | 96 | 8 | `-91.67%` |
| 平均等效半径 | `8.3884 nm` | `14.8713 nm` | `+77.28%` |
| 平均 `R^3` | `702.84 nm^3` | `5888.40 nm^3` | `+737.6%` |
| beta 体积分数 | `0.0239295` | `0.0248527` | `+3.86%` |
| 界面面积代理 | `0.0075493` | `0.0032282` | `-57.24%` |
| 矩阵 `x_Ag` | `0.0062007` | `0.0052645` | `-15.10%` |

该结果支持：6 h之后进入明显的后成核粗化阶段，表现为粒子数量下降、
幸存粒子长大、平均界面面积下降，而总 beta 体积分数仅缓慢变化。

从12 h到48 h的子区间同样显示：粒子数 `34 -> 8`，平均半径
`11.0384 -> 14.8713 nm`，平均 `R^3` 增长约 `225.7%`。

## 5. 失败/警告审计

运行退出码为0，stderr没有运行时错误。保留的两条警告为：

1. `gp_birth_enabled` 是旧参数文件中的未知键并被忽略；本运行 GP 路径
   明确关闭，因此没有改变本次无弹性结果。
2. `lambda_sm/dx = 4` 处于界面分辨率建议阈值边界；这是分辨率提示，
   不是数值崩溃或守恒失败。

## 6. 证据位置

远端运行根目录：

```text
/data/home/luozhiheng/tmp/pf_random_psd_v2_246x96_continuation_12h48h_20260730
```

关键文件：

```text
slurm-72831.out
slurm-72831.err
checkpoint_12h_to_48h.chk
pf_input.params
main_cuda
```

本地趋势数据：

```text
reports/pf_6h_48h_coarsening_random_psd_v2/population_matrix_time_series_exploratory.csv
```

## 7. 最终使用边界

本报告可以作为当前课题的正式无弹性6–48 h趋势证据，支持后成核粗化
方向的判断。涉及“当前冻结源码二进制完全可复现”的生产资格声明时，
仍必须将二进制哈希差异写入 provenance，并可由作业72915完成同源复验。

```text
recommended_use=FORMAL_TREND_EVIDENCE_WITH_EXPLICIT_PROVENANCE
not_claimed=FROZEN_BINARY_SOURCE_MATCH
```
