# Critical Radius And Continue-Dynamics Workflow

这份文档总结当前仓库里一条完整、可复用的流程：

1. 用 Schur 预测脚本先估算不同外部应变下的临界半径。
2. 生成每个 case 的半径扫描 `case_*.csv`。
3. 用 Slurm 脚本在 cluster 上做 PF constraint/minimize 扫描，得到 PF 版本的临界半径。
4. 用 summary/energy 自动汇总脚本提取：
   - 离散峰位 `rc_cnt_nm`
   - 局部三次/四次拟合峰位 `rc_cnt_fit_nm`
   - 峰值对应形貌 summary
5. 再基于这张总表，为每个 case 自动生成 `dynamics-continue` 任务：
   - 不再直接用峰位 case 的 VTK
   - 而是读取“比峰位更大的下一个离散半径 case”的 `phi_final/xB_final`
   - 再提交到 `24h` 和 `uvip` 两个队列继续跑核子生长

下面所有命令默认从仓库根目录运行：

`/Users/heng/Documents/GitHub/CUDA_STO_PF`

cluster 对应目录是：

`/data/home/luozhiheng/CUDA_STO_PF`

## 1. 用 Schur 脚本预测临界半径

脚本：

- [`compute_schur_rc_predictions.py`](/Users/heng/Documents/GitHub/CUDA_STO_PF/tools/analysis/compute_schur_rc_predictions.py)

作用：

- 在 strictref 语义下计算 `exx / eyy / exx+eyy` 外载对应的 Schur 预测临界半径
- 同时输出每个 case 对应的 `case_*.csv`

典型命令：

```bash
python3 tools/analysis/compute_schur_rc_predictions.py \
  --input-json physical_inputs.example.json \
  --output-dir Results_scan/constraint_cnt_strictref_test_T400_x0_0p03 \
  --xB-out 0.03 \
  --window-nm 0.5 \
  --step-nm 0.1 \
  --strains 0,0.0025,0.005,0.0075,0.01 \
  --modes exx,eyy,exx_eyy
```

输出：

- `schur_rc_predictions.csv`
- 一系列 `case_*.csv`

例如：

- `Results_scan/constraint_cnt_strictref_test_T400_x0_0p03/case_exx_s0p01_L5.csv`
- `Results_scan/constraint_cnt_strictref_test_T400_x0_0p03/case_eyy_sm0p005_L5.csv`

## 2. 用 Slurm 在 cluster 上做 PF 临界半径扫描

PF constraint/minimize 扫描当前使用的 cluster 脚本是：

- `/data/home/luozhiheng/CUDA_STO_PF/jobs/submit_constraint_cnt_case_sweep.sbatch`

说明：

- 一条 Slurm array task 对应一个 `base_case_tag`
- 该脚本会在 `rc_schur_nm ± window_nm` 的窗口内，用 `step_nm` 做离散半径扫描
- 每个离散半径点都会运行一次 full-model minimize
- 当前这一步的结果会写入 `Results/chel_.../cntcon_.../`

单个 case 的提交示例：

```bash
ssh uvip-cluster '
cd /data/home/luozhiheng/CUDA_STO_PF &&
sbatch --array=0-0 jobs/submit_constraint_cnt_case_sweep.sbatch \
  /data/home/luozhiheng/CUDA_STO_PF/Results_scan/constraint_cnt_strictref_test_T400_x0_0p03/case_exx_s0p01_L5.csv
'
```

一整批 case 的思路：

- `case_*.csv`：适合一个 mode/strain 一组地提交
- `cases_batch_*.csv`：适合按优先级或正负载分批提交

这一步结束后，每个离散半径点都会在 `Results/` 下形成一个 case 子目录，例如：

- `/data/home/luozhiheng/CUDA_STO_PF/Results/chel_T400_cuda_400x400x400_dt0.1_steps30000_r2.464nm_xB0.030/cntcon_T400_xB0p030_strictref_exx_s000_r2p464406`

其中通常包含：

- `energy_minimize_*.csv`
- `phi_final_*.vtk`
- `xB_final_*.vtk`
- `summary_*.txt`

## 3. 汇总 PF 临界半径与峰值形貌

汇总脚本：

- [`analyze_cnt_peak_table.py`](/Users/heng/Documents/GitHub/CUDA_STO_PF/tools/analysis/analyze_cnt_peak_table.py)

作用：

- 读取一整个 case 目录下的离散半径扫描结果
- 从 `energy_minimize_*.csv` 中提取 `F_total_CNT_hat`
- 先取离散峰值 `rc_cnt_nm`
- 再在峰值附近做局部 cubic/quartic 拟合，得到 `rc_cnt_fit_nm`
- 读取峰值对应的 `summary_*.txt`
- 把形貌信息也并到总表中

典型命令：

```bash
python3 tools/analysis/analyze_cnt_peak_table.py \
  --case-dir Results_scan/constraint_cnt_strictref_test_T400_x0_0p03 \
  --results-root /data/home/luozhiheng/CUDA_STO_PF \
  --output Results_scan/constraint_cnt_strictref_test_T400_x0_0p03/current_results_master_table_fitted.csv
```

输出总表：

- `current_results_master_table_fitted.csv`

这个表里已经包含：

- `rc_schur_nm`
- `rc_cnt_nm`
- `rc_cnt_fit_nm`
- `F_CNT_peak_hat`
- `summary_path`
- `L1/L2/L3`
- `L1_over_L3`
- 主轴方向与法向信息

## 4. 从总表自动生成 continue-dynamics 任务

脚本：

- [`prepare_dynamics_continue_from_summary.py`](/Users/heng/Documents/GitHub/CUDA_STO_PF/tools/analysis/prepare_dynamics_continue_from_summary.py)
- [`submit_dynamics_continue_batch.py`](/Users/heng/Documents/GitHub/CUDA_STO_PF/tools/analysis/submit_dynamics_continue_batch.py)

### 4.1 单个 case 生成 continue 命令

可以先对某一行做检查：

```bash
python3 tools/analysis/prepare_dynamics_continue_from_summary.py \
  --csv Results_scan/constraint_cnt_strictref_test_T400_x0_0p03/current_results_master_table_fitted.csv \
  --case-tag cntcon_T400_xB0p030_strictref_exx_s000 \
  --emit manifest \
  --template sbatch
```

这一步会做的事情：

- 读取 `current_results_master_table_fitted.csv`
- 找到该 case 的离散峰位 `rc_cnt_nm`
- 不再用峰位 case 自己的 `phi_final/xB_final`
- 而是去找“下一个更大的离散半径 case”的：
  - `phi_final_*.vtk`
  - `xB_final_*.vtk`

也就是说，continue 的真实初始场是：

- **更大离散半径 case 的最终核子场**

而不是：

- 峰位 case 的最终场 + 只改半径标签

### 4.2 整批提交 continue dynamics

批量提交脚本：

- [`submit_dynamics_continue_batch.py`](/Users/heng/Documents/GitHub/CUDA_STO_PF/tools/analysis/submit_dynamics_continue_batch.py)

典型命令：

```bash
ssh uvip-cluster '
cd /data/home/luozhiheng/CUDA_STO_PF &&
python3 tools/analysis/submit_dynamics_continue_batch.py \
  --csv /data/home/luozhiheng/CUDA_STO_PF/Results_scan/constraint_cnt_strictref_test_T400_x0_0p03/current_results_master_table_fitted.csv \
  --results-root /data/home/luozhiheng/CUDA_STO_PF \
  --extra-steps 5000 \
  --radius-mode supercritical \
  --radius-offset-nm 0.1 \
  --start-queue 24h
'
```

这会自动：

- 把完成的 case 分配到 `gpu_vip_24h` 和 `gpu_uvip`
- 为两个队列分别写 manifest：
  - `gpu_vip_24h_dynamics_continue_manifest.csv`
  - `gpu_uvip_dynamics_continue_manifest.csv`
- 自动提交两个串行 Slurm job

当前 continue 使用的 Slurm 模板：

- [`submit_continue_manifest_serial.sbatch`](/Users/heng/Documents/GitHub/CUDA_STO_PF/jobs/submit_continue_manifest_serial.sbatch)

## 5. continue dynamic 与 minimize continue 的目录命名

当前已经统一成：

- `dynamics-continue`
  - `continue_dyn_1`
  - `continue_dyn_2`
- `minimize-continue`
  - `continue_min_1`
  - `continue_min_2`

这样两类 continue 结果不会再混在一起。

同时，continue 的输出也不再写在 run-root 根目录，而是统一写进子目录。

例如：

- 原始离散 PF case：
  - `.../chel_T400_..._r2.936nm_xB0.030/cntcon_T400_..._r2p935828/`
- continue dynamic：
  - `.../chel_T400_..._r2.936nm_xB0.030/continue_dyn_1/`

所以原始离散扫描结果和 continue 生长结果是分开的。

## 6. continue 完成后再做生长汇总

continue 生长后汇总脚本：

- [`summarize_dynamics_continue_growth.py`](/Users/heng/Documents/GitHub/CUDA_STO_PF/tools/analysis/summarize_dynamics_continue_growth.py)

这一步可以把：

- `start_radius_nm`
- `grown_equiv_radius_nm`
- `delta_growth_nm`
- `L1/L2/L3`
- `L1_over_L3`

汇成新的 continue 生长总表。

## 7. 推荐的实际执行顺序

建议按这个顺序跑：

1. 本地运行 `compute_schur_rc_predictions.py`
2. 得到 `case_*.csv`
3. 在 cluster 上用 `submit_constraint_cnt_case_sweep.sbatch` 做 PF 临界半径扫描
4. 本地或 cluster 上运行 `analyze_cnt_peak_table.py`
5. 生成 `current_results_master_table_fitted.csv`
6. 用 `submit_dynamics_continue_batch.py` 自动生成并提交 continue dynamic
7. 跑完后再用 `summarize_dynamics_continue_growth.py` 做 continue 生长汇总

## 8. 当前这一版 workflow 的关键约定

当前版本最重要的约定有两条：

1. `dynamics-continue` 不是从拟合峰位生成新核子

因为拟合峰位 `rc_cnt_fit_nm` 没有对应的独立 VTK，所以 continue 初始场不能直接来自拟合峰位。

2. `dynamics-continue` 用的是“比离散峰位更大的下一个离散 case”

也就是：

- 先用 PF 扫描找到 `rc_cnt_nm`
- 再选择扫描窗口里**下一个更大的离散半径点**
- 用那个 case 的 `phi_final/xB_final` 继续跑动力学生长

这个做法的优点是：

- 初始场是真实存在的 PF 收敛场
- 不是只改标签
- 更适合测试“临界之后是否长大”

