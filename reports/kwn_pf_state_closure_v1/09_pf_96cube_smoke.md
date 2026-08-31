# 96³ PF smoke matrix

## Gate

`PF_96CUBE_SMOKE_GATE=NOT_RUN_NO_CUDA_OR_PF_BINARY`。

本轮只面向 `pf_mass_conserving_library_handoff_six_particle_96cube_v1`。当前工作站没有可用于本合同的 `nvcc` 和可运行 PF binary，因而没有启动 CUDA PF，不能把 host controls 写成 A–E trajectory。

| case | host-side 准备/证据 | runtime adapter | actual CUDA 状态 |
|---|---|---|---|
| A `BASELINE_LEGACY_ZERO_AUX` | 无 auxiliary 的基线定义 | 不需要 sidecar | `NOT_RUN_NO_CUDA_OR_PF_BINARY` |
| B `IDENTITY_ADAPTER_ZERO_AUX` | zero-aux mapping 与 V6 package-bound checkpoint host control | `SOURCE_INTEGRATED_UNCOMPILED_NOT_RUN` | `NOT_RUN_NO_CUDA_OR_PF_BINARY` |
| C `NONZERO_FROZEN_AUX_STORAGE` | nonzero four-bucket storage 与 compact PSD persistence host control | `SOURCE_INTEGRATED_UNCOMPILED_NOT_RUN` | `NOT_RUN_NO_CUDA_OR_PF_BINARY` |
| D `CONSERVATIVE_MATRIX_TO_GP_CONTROL` | exact inverse-storage roundtrip host control | `SOURCE_INTEGRATED_UNCOMPILED_NOT_RUN` | `NOT_RUN_NO_CUDA_OR_PF_BINARY` |
| E `FIXTURE_CONDITIONED_KWN_HANDOFF_V2` | real host-profile package/readback feasibility control与 raw-init materialization | `SOURCE_INTEGRATED_UNCOMPILED_NOT_RUN` | `NOT_RUN_NO_CUDA_OR_PF_BINARY` |

host controls 的非动力学证据包括：zero-aux reconstructed matrix `xB` 最大差 `4.336808689942018e-18`；S2 frozen nonzero storage 保留 source local fields；S3 reverse map 回到 `max |ΔxB|=2.6020852139652106e-18`。这些只说明初态/存储映射，不是 A–E 的 6 h 或 48 h PF 观测。

因此没有测量任何 CUDA case 的四库存时间序列、`xB_matrix` mean/min/max、`f_beta`、粒子数、平均半径、`S_v`、free-energy components、clipping、NaN/Inf、restart hash 或 GP/subgrid PSD checksum。matrix pulse、seed loss、energy jump、profile relaxation time 与质量漂移均为 `NOT_MEASURED_NO_CUDA`，而不是通过或数值失败。

`scripts/write_pf_smoke_not_run_status.py` 在确认运行时不可用时写出 `outputs/kwn_pf_state_closure_v1/pf_smoke_trajectories.csv`：A–E × 0/6/48 h 的每个数值单元为空，并标为 `NOT_A_PF_TRAJECTORY`。该 CSV 是缺失运行的诚实状态工件，不是 PF 轨迹。
