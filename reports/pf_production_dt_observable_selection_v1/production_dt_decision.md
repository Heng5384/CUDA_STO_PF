# 主要物理观察量生产 dt 收敛选择

## 最终决定

```text
production_dt_observable_convergence_status=PASS_PRODUCTION_DT_OBSERVABLE_CONVERGENCE_V1
selected_production_dt_code=0.02
selected_production_dt_physical_s=0.9909260953431841
fallback_dt_code=0.01
restart_bytewise_equal=true
final_status=PASS_PRODUCTION_DT_OBSERVABLE_CONVERGENCE_V1
```

生产默认时间步选择为：

\[
dt_{\rm code}=0.02,\qquad
dt_{\rm physical}=0.9909260953431841\ {\rm s/step}.
\]

回退时间步为：

\[
dt_{\rm code}=0.01,\qquad
dt_{\rm physical}=0.4954630476715921\ {\rm s/step}.
\]

本决定只适用于当前冻结的380 °C、\(dx=1\) nm、
\(\lambda_{\rm sm}=4\) nm、有弹性、PF-only条件接管路径。它不是完整
6–48 h轨迹的实验验证，也不授权重新加入GP、GP Birth、GP release或
新β成核。

## 验收设计

使用同一份hash-pinned六粒子96³条件接管fixture，从6 h起运行到相同
终点：

```text
common_code_time=5.12
common_physical_time_s=253.67708040785513
common_endpoint_age_h=6.070465855668848
```

三档时间步为：

| 案例 | dt_code | 物理步长 (s) | 步数 | 墙钟时间 (s) |
|---|---:|---:|---:|---:|
| dt0p02 | 0.02 | 0.9909260953431841 | 256 | 16.331 |
| dt0p01 | 0.01 | 0.4954630476715921 | 512 | 32.275 |
| dt0p005 | 0.005 | 0.2477315238357960 | 1024 | 64.344 |

以最细的dt=0.005为参考。所有案例均保持6个粒子，身份映射精确，
无merge/split，stderr为空，守恒零模通过，GP、外部源和新β成核均未启用。

## dt=0.02相对dt=0.005的误差

| 主要观察量 | 误差 | 硬门槛 | 结果 |
|---|---:|---:|---|
| β体积分数 | 0.05258% | 1% | PASS |
| 平均等效半径 | 0.02302% | 1% | PASS |
| \(S_v\) | 0.04034% | 2% | PASS |
| \(M_6\) | 0.06638% | 5% | PASS |
| 远场matrix \(x_{\rm Ag}\) | \(1.1908\times10^{-5}\)绝对差 | \(4\times10^{-4}\) | PASS |
| 平均弹性能 | 0.06519% | 5% | PASS |
| canonical inventory | \(1.6050\times10^{-13}\)相对误差 | \(10^{-10}\) | PASS |
| 全场\(\phi\) MAE | \(3.9233\times10^{-5}\) | 归一化后2% | PASS |

全场\(x_B\) MAE为 \(5.3888\times10^{-5}\)，按照已冻结的条件接管
政策保留为时间离散诊断量，但不单独否决主要物理观察量收敛。

## 数值边界审计

第一次自动分析误把\(\phi\)限定为严格的\([0,1]\)，因此三档（包括最细
参考档）同时得到`bounds=false`。原始场审计表明：

```text
phi_min=-1.0e-6
phi_max<=0.999303
xB_min>0
xB_max<1
```

CUDA求解器的既有数值合同明确允许
\(\phi\in[-10^{-6},1+10^{-6}]\)。分析器随后改为使用该源码合同，并把
实际min/max写入机器可读审计；没有修改求解器、物理参数或运行数据。
原始阻塞审计仍保留在workstation输出根中。

## restart

dt=0.02连续256步与128步checkpoint后restart至256步的最终checkpoint
逐字节一致：

```text
continuous_checkpoint_sha256=148bd979e09047907f7b30f1f8904b5569a4f201d556750293f6f8f629085d9c
restart_checkpoint_sha256=148bd979e09047907f7b30f1f8904b5569a4f201d556750293f6f8f629085d9c
```

## Provenance

```text
workstation_root=/home/zhiheng/tmp/pf_production_dt_observable_selection_v2_20260731
fixture_manifest_sha256=a87e76405bd1b901b6848b6c39d788ff3fe537ca0acea93f99b64ac67e6ced81
profile_library_manifest_sha256=58803a8bc6679b823e45e7a7b85df16ae68efa55338d52d4c4151b414a5ef0fe
binary_sha256=7581c169fb1d1c16ee60764f418ee9c33508682c8b9dd8cf76b89368c7990602
analysis_script_sha256=2eacc7b836c4f4ab463b752c26e9d3f972afda1ba5c75611f103c2f0063dfc8d
runner_sha256=9562b671833a96711406021dba5b8ce8cd3c6bd1828a0e7b2909dd06dd4eb898
final_audit_sha256=234fece90c5a0b94a252ed6f6af6b07c9c611ef9f0c08f2b8765ded327fccede
final_terminal_sha256=df7524056dc38eab95d88001cfefd602d878e12b204015c019c1902dc51ecb65
cluster_used=false
commit_created=false
push_performed=false
```

