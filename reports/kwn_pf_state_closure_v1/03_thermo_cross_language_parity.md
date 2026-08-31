# 跨语言热力学一致性

## 结果

`PASS_THERMO_VALIDATION_HASH_BIND`（host-side thermo probe，不是完整 CUDA PF run）。

| 项目 | 结果 |
|---|---:|
| validation contract hash | `d0ff02973ab0f737043e1a40d4f69893a469cbfe2bc4cd22f9e6a410bd0b1333` |
| 温度 | 653.15 K |
| `xB` 点数 | 20（最大 0.2） |
| 半径 | 2、5、10、20、50 nm |
| 比较行数 | 100 |
| convex extrapolation | `enabled=false`，probe 同时检查 wrapper 输出开关与合同相等 |
| 最大相对误差 | `3.657021380996818e-12` |
| planar solvus 最大绝对误差 | `0.0` |
| solvus 以下 beta driving force | `-3627.797031279093` J mol⁻¹ |
| solvus 以上 beta driving force | `3492.349623820144` J mol⁻¹ |

通过阈值为基础量 `relative error <= 1e-10`、solvus `absolute x error <= 1e-8`，并要求溶解/生长方向分别为负/正。上述结果满足这三项条件。

## 比较内容与实现

Python contract 与临时编译的 C++ probe 逐点比较：`G_alpha`、两端 chemical potential、`dG/dx`、`d²G/dx²`、beta driving force、`D(T)`、planar solvus、curvature-corrected equilibrium 与 alpha/beta molar-volume conversion。C++ probe 使用 `mu_PbTe_raw`、`mu_Ag2Te_raw` 及其 raw derivative wrapper，而不是绕开活动 `thermo_utils.h` 路径的独立 CALPHAD 重算。

扩大到 `xB=0.08, 0.09, 0.1, 0.2` 的目的，是确认生成的 convex 开关确为关闭状态时，PF host wrapper 与 Python contract 在该活动分支上保持同 hash 一致；它不把这组 host 比较升级为完整 PF composition trajectory。

逐点表：`outputs/kwn_pf_state_closure_v1/thermo_cross_language.csv`；摘要：`outputs/kwn_pf_state_closure_v1/thermo_cross_language_summary.json`。这项 gate 不包含 CUDA kernel、弹性求解或 0/6/48 h PF dynamics。
