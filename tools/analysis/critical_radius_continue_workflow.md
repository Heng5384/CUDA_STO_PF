# CNT 临界半径与 Continue-Dynamic 标准工作流

这套流程的目标是把一次完整任务规范成一组固定步骤：

1. 先做 Schur 预测，生成一份引导表。
2. 用一条串行 sbatch 直接读取引导表，完成整批 PF 临界半径扫描。
3. 扫描结束后，用同一份引导表生成 CNT 临界半径汇总。
4. 再根据临界半径汇总，自动生成 continue dynamic 引导表。
5. 用一条串行 sbatch 读取 continue 引导表，继续跑核子生长。
6. 最后对 continue 结果做统一汇总。

这套流程只要求你调整两个物理输入：

- `temperature_C`
- `xB_out`

其余的路径组织、输入表、汇总表和 continue 输入选择都按统一约定自动处理。

## 目录约定

每一个大任务都放在：

```text
Results/workflows/T400_xB0p030/
```

目录结构固定为：

```text
Results/workflows/T400_xB0p030/
├── input/
│   ├── physical_inputs.json
│   ├── workflow_meta.json
│   ├── schur_rc_predictions.csv
│   ├── guide_cnt_scan.csv
│   └── guide_continue_dynamic.csv
├── raw/
│   └── chel_T.../cntcon_.../
├── cnt_scan/
│   └── current_results_master_table_fitted.csv
├── continue_dynamic/
│   └── growth_summary.csv
└── summaries/
```

说明：

- `input/` 只放引导 CSV 和工作流配置。
- `raw/` 放主程序实际输出的原始结果。
- `cnt_scan/` 放临界半径汇总。
- `continue_dynamic/` 放 continue 生长汇总。

这样不同温度、不同 `xB_out` 的任务天然分开，不会混在一个目录里。

## 迁移到别人的 Cluster 账户

这套流程支持直接 `git pull` 到别人的 cluster 账户下使用，但文档里的**示例命令**默认写的是当前账户环境：

- 仓库根目录：
  - `/data/home/luozhiheng/CUDA_STO_PF`
- 队列示例：
  - `gpu_uvip / gpu_uvip`

如果换到别人的 cluster 账户，通常**不需要改 Python 脚本本身**，只需要改这 3 个外部变量：

- `REPO_ROOT`
  - 别人实际 `git clone` 或 `git pull` 后的仓库绝对路径
- `QUEUE`
  - 目标 cluster 可用的 partition 名称
- `QOS`
  - 对应 queue 的 qos 名称

推荐在命令前先显式设这 3 个变量，再照抄后续命令：

```bash
export REPO_ROOT=/data/home/<other_user>/CUDA_STO_PF
export QUEUE=gpu_uvip
export QOS=gpu_uvip
```

然后所有命令都统一写成：

```bash
cd "${REPO_ROOT}"
```

和：

```bash
GUIDE_CSV="${REPO_ROOT}/Results/workflows/T400_xB0p030/input/guide_cnt_scan.csv" \
sbatch -p "${QUEUE}" --qos="${QOS}" jobs/submit_cnt_guide_serial.sbatch
```

### 需要提醒阅读者修改的地方

如果这份文档是给别人看的，最人性化的提示方式不是写“请自行修改路径”，而是明确告诉他只改下面 3 项：

```text
在开始之前，请先确认并替换：
1. REPO_ROOT：你的仓库绝对路径
2. QUEUE：你要提交到的 partition
3. QOS：对应的 qos
其余命令保持不变即可
```

### 哪些地方通常不用改

下面这些通常不需要因为换账户而改代码：

- `tools/analysis/*.py`
  - 这些脚本主要通过命令行参数和相对路径工作
- `jobs/submit_cnt_guide_serial.sbatch`
- `jobs/submit_continue_dynamic_guide_serial.sbatch`

这两个 sbatch 脚本会：

- 自动定位仓库根目录
- 读取 `GUIDE_CSV`
- 按 guide 里的相对路径组织输出

所以换账户后，真正需要变的是：

- 你在 shell 里给的 `GUIDE_CSV`
- `sbatch -p ... --qos ...`
- 仓库所在的绝对路径

### 什么时候才需要改代码

只有在下面这些条件变化时，才建议改脚本或文档默认值：

- 对方 cluster 的 queue/qos 规则和当前完全不同
- 对方的 `sbatch` 资源申请模板不同
- 对方要求不同的默认 `time / gres / mem`
- 对方没有把仓库放在标准 Linux 路径下

这种情况下，优先改的是：

- `jobs/submit_cnt_guide_serial.sbatch`
- `jobs/submit_continue_dynamic_guide_serial.sbatch`

而不是流程 Python 脚本。

如果对方还没有这个私有仓库的访问权限，先看：

- [`cluster_git_access_guide.md`](/Users/heng/Documents/GitHub/CUDA_STO_PF/tools/analysis/cluster_git_access_guide.md)

## 关键脚本

### 1. 建立工作流目录并生成引导表

脚本：

- [`setup_cnt_workflow.py`](/Users/heng/Documents/GitHub/CUDA_STO_PF/tools/analysis/setup_cnt_workflow.py)

作用：

- 读取基础物理输入 JSON。
- 覆盖指定的温度和 `xB_out`。
- 计算 Schur 预测临界半径。
- 生成：
  - `input/physical_inputs.json`
  - `input/workflow_meta.json`
  - `input/schur_rc_predictions.csv`
  - `input/guide_cnt_scan.csv`

零应变去重规则：

- 当应变为 `0.0` 时，只生成一条基准 case：
  - `exx, s000`
- 不再重复生成：
  - `eyy, s000`
  - `exx_eyy, s000`

原因：

- 在外部应变为零时，这三种加载方式对 PF 扫描来说是同一个基准状态
- 保留一条即可，避免引导 CSV 和后续扫描结果出现重复的零应变 case

### 2. 读取 CNT 扫描引导表并串行提交

脚本：

- [`submit_cnt_guide_serial.sbatch`](/Users/heng/Documents/GitHub/CUDA_STO_PF/jobs/submit_cnt_guide_serial.sbatch)

作用：

- 读取一份 `guide_cnt_scan.csv`
- 串行跑完整批 CNT 半径扫描
- 不拆很多子任务
- 只提交到一个队列
- 如果同一份 guide 被重复提交，会自动跳过已经完成的 case，只继续未完成部分

### 3. 从引导表和原始结果生成 CNT 汇总

脚本：

- [`summarize_cnt_scan_from_guide.py`](/Users/heng/Documents/GitHub/CUDA_STO_PF/tools/analysis/summarize_cnt_scan_from_guide.py)

作用：

- 仍然读取 `guide_cnt_scan.csv`
- 自动扫描对应 `raw/` 里的结果
- 提取：
  - `rc_schur_nm`
  - `rc_cnt_nm`
  - `rc_cnt_fit_nm`
  - `F_CNT_peak_hat`
  - 峰值对应形貌 summary

输出：

- `cnt_scan/current_results_master_table_fitted.csv`

### 4. 从 CNT 汇总生成 continue dynamic 引导表

脚本：

- [`prepare_continue_dynamic_guide.py`](/Users/heng/Documents/GitHub/CUDA_STO_PF/tools/analysis/prepare_continue_dynamic_guide.py)

作用：

- 读取：
  - `cnt_scan/current_results_master_table_fitted.csv`
  - `input/guide_cnt_scan.csv`
- 按规则选择 continue 初始核子：
  - 不是直接用拟合峰位
  - 也不是只改半径标签
  - 而是选取“**大于离散峰位指定偏移量**”的离散半径 case
  - 读取那个 case 的：
    - `phi_final_*.vtk`
    - `xB_final_*.vtk`

输出：

- `input/guide_continue_dynamic.csv`

### 5. 读取 continue 引导表并串行提交

脚本：

- [`submit_continue_dynamic_guide_serial.sbatch`](/Users/heng/Documents/GitHub/CUDA_STO_PF/jobs/submit_continue_dynamic_guide_serial.sbatch)

作用：

- 读取一份 `guide_continue_dynamic.csv`
- 串行跑完整批 `dynamics-continue`
- 只提交到一个队列
- 如果同一份 guide 被重复提交，会自动跳过已经完成的 continue case，只继续未完成部分

说明：

- continue 时会以源 case 的 `pf_input.params` 为底
- 只覆盖：
  - `dt`
- 其余 continue 物理参数保持不变

如果需要跑 `minimize-continue`，对应脚本是：

- [`submit_continue_minimize_guide_serial.sbatch`](/Users/heng/Documents/GitHub/CUDA_STO_PF/jobs/submit_continue_minimize_guide_serial.sbatch)

它和 `dynamics-continue` 一样，也支持：

- 重复提交同一份 guide 时自动跳过已完成 row
- 只继续未完成部分

### 6. continue 生长汇总

脚本：

- [`summarize_continue_from_guide.py`](/Users/heng/Documents/GitHub/CUDA_STO_PF/tools/analysis/summarize_continue_from_guide.py)

作用：

- 读取 `guide_continue_dynamic.csv`
- 扫描对应 continue 结果
- 汇总：
  - `start_radius_nm`
  - `grown_equiv_radius_nm`
  - `delta_growth_nm`
  - `voxel_count`
  - `L1/L2/L3`
  - `L1_over_L3`

输出：

- `continue_dynamic/growth_summary.csv`

### 7. guide 进度检查与断点续跑

脚本：

- [`report_guide_progress.py`](/Users/heng/Documents/GitHub/CUDA_STO_PF/tools/analysis/report_guide_progress.py)

作用：

- 读取一份 guide CSV
- 检查每一行对应结果是否已经完成
- 输出一张进度表
- 可选地再导出一份只包含未完成 row 的 `pending-only guide`

支持三种 guide 类型：

- `cnt`
- `continue-dynamic`
- `continue-minimize`

#### 推荐理解方式

这套 workflow 的“续跑”机制有两层：

1. **最简单模式**
- 直接重复提交原始 guide
- sbatch 脚本会自动跳过已完成 row
- 只继续未完成 row

2. **显式检查模式**
- 先用 `report_guide_progress.py` 生成进度表
- 再导出一份 `pending-only guide`
- 然后只提交这份 pending guide

两种方式都可以。

#### CNT 扫描进度检查

```bash
cd "${REPO_ROOT}"

python3 tools/analysis/report_guide_progress.py \
  --guide-csv Results/workflows/T400_xB0p030/input/guide_cnt_scan.csv \
  --repo-root "${REPO_ROOT}" \
  --guide-type cnt \
  --pending-guide-output Results/workflows/T400_xB0p030/input/guide_cnt_scan_pending.csv
```

输出：

- `guide_cnt_scan_progress.csv`
- `guide_cnt_scan_pending.csv`

#### continue dynamic 进度检查

```bash
cd "${REPO_ROOT}"

python3 tools/analysis/report_guide_progress.py \
  --guide-csv Results/workflows/T400_xB0p030/input/guide_continue_dynamic.csv \
  --repo-root "${REPO_ROOT}" \
  --guide-type continue-dynamic \
  --pending-guide-output Results/workflows/T400_xB0p030/input/guide_continue_dynamic_pending.csv
```

#### continue minimize 进度检查

```bash
cd "${REPO_ROOT}"

python3 tools/analysis/report_guide_progress.py \
  --guide-csv Results/workflows/T400_xB0p030/input/guide_continue_dynamic.csv \
  --repo-root "${REPO_ROOT}" \
  --guide-type continue-minimize \
  --pending-guide-output Results/workflows/T400_xB0p030/input/guide_continue_minimize_pending.csv
```

#### 什么叫“已完成”

CNT 扫描：

- 已有 `energy_minimize_*.csv`
- 且已有 `phi_final_*.vtk`

continue dynamic：

- 已有 `continue_dyn_1/summary.txt`
- 或已有最终步的 `phi_<nsteps>.vtk` 和 `xB_<nsteps>.vtk`

continue minimize：

- 已有 `continue_min_1/summary_continue_min_1.txt`
- 或已有 `energy_minimize_continue_min_1.csv` 和 `phi_final_continue_min_1.vtk`

#### 什么时候用 pending-only guide

如果你只是想继续跑完当前任务，最省事的是：

- 直接重新提交原 guide

如果你想：

- 明确知道还剩哪些 row
- 只提交没完成的部分
- 保存一份可审计的“待续跑清单”

那就先生成 `pending-only guide` 再提交。

## 单队列使用方式

这套流程不再拆成两个队列。

你只需要手动选择一个队列，例如：

- `gpu_uvip / gpu_uvip`
或
- `gpu_vip_24h / gpu_vip_24h`

提交时直接用：

```bash
sbatch -p gpu_uvip --qos=gpu_uvip jobs/submit_cnt_guide_serial.sbatch
```

或者：

```bash
sbatch -p gpu_vip_24h --qos=gpu_vip_24h jobs/submit_cnt_guide_serial.sbatch
```

continue 也是同样的单队列提法。

## 推荐参数约定

默认使用：

- 网格：`400,400,400`
- CNT 扫描步数：`30000`
- CNT 扫描 `dt`：`0.1`
- CNT 扫描输出间隔：`250`
- continue dynamic 步数：`5000`
- continue dynamic `dt`：`0.1`
- continue dynamic 输出间隔：`2500`
- continue 起始半径规则：
  - `rc_cnt_nm + 0.1 nm`

如果后续需要改，可以通过脚本参数显式覆盖。

## 最短命令清单

下面这套命令假设：

- 仓库根目录：
  - `${REPO_ROOT}`
- 温度：
  - `400`
- `xB_out`：
  - `0.03`
- 队列：
  - `${QUEUE}`
- QOS：
  - `${QOS}`

先设变量：

```bash
export REPO_ROOT=/data/home/luozhiheng/CUDA_STO_PF
export QUEUE=gpu_uvip
export QOS=gpu_uvip
```

### A. 建立工作流目录并生成 CNT 引导表

```bash
cd "${REPO_ROOT}"

python3 tools/analysis/setup_cnt_workflow.py \
  --temp-c 400 \
  --xb-out 0.03 \
  --lambda-sm-nm 0.6 \
  --base-json physical_inputs.example.json \
  --workflow-root Results/workflows \
  --grid 400,400,400 \
  --dt 0.03 \
  --steps 30000 \
  --out-every 2500 \
  --csv-out-every 10 \
  --window-nm 0.5 \
  --step-nm 0.1
```

这一步会生成：

```text
Results/workflows/T400_xB0p030/input/guide_cnt_scan.csv
```

### B. 用 guide_cnt_scan.csv 提交整批 CNT 扫描

```bash
cd "${REPO_ROOT}"

GUIDE_CSV="${REPO_ROOT}/Results/workflows/T400_xB0p030/input/guide_cnt_scan.csv" \
sbatch -p "${QUEUE}" --qos "${QOS}" jobs/submit_cnt_guide_serial.sbatch
```

raw 结果会自动写到：

```text
Results/workflows/T400_xB0p030/raw/
```

### C. 生成 CNT 临界半径汇总

```bash
cd "${REPO_ROOT}"

python3 tools/analysis/summarize_cnt_scan_from_guide.py \
  --guide-csv Results/workflows/T400_xB0p030/input/guide_cnt_scan.csv \
  --repo-root "${REPO_ROOT}"
```

输出：

```text
Results/workflows/T400_xB0p030/cnt_scan/current_results_master_table_fitted.csv
```

### D. 根据 CNT 汇总生成 continue dynamic 引导表

```bash
cd "${REPO_ROOT}"

python3 tools/analysis/prepare_continue_dynamic_guide.py \
  --summary-csv Results/workflows/T400_xB0p030/cnt_scan/current_results_master_table_fitted.csv \
  --guide-csv Results/workflows/T400_xB0p030/input/guide_cnt_scan.csv \
  --repo-root "${REPO_ROOT}" \
  --radius-offset-nm 0.1 \
  --dt 0.1 \
  --steps 5000 \
  --out-every 2500 \
  --csv-out-every 10
```

输出：

```text
Results/workflows/T400_xB0p030/input/guide_continue_dynamic.csv
```

### E. 用 guide_continue_dynamic.csv 提交整批 continue dynamic

```bash
cd "${REPO_ROOT}"

GUIDE_CSV="${REPO_ROOT}/Results/workflows/T400_xB0p030/input/guide_continue_dynamic.csv" \
sbatch -p "${QUEUE}" --qos "${QOS}" jobs/submit_continue_dynamic_guide_serial.sbatch
```

continue 结果会落到：

```text
Results/workflows/T400_xB0p030/raw/.../continue_dyn_1/
```

### F. 生成 continue 生长汇总

```bash
cd "${REPO_ROOT}"

python3 tools/analysis/summarize_continue_from_guide.py \
  --guide-csv Results/workflows/T400_xB0p030/input/guide_continue_dynamic.csv \
  --repo-root "${REPO_ROOT}"
```

输出：

```text
Results/workflows/T400_xB0p030/continue_dynamic/growth_summary.csv
```

## 实际操作原则

后续人工操作时，建议主要改这两个输入：

- `--temp-c`
- `--xb-out`

如果需要切换界面宽度，再额外改：

- `--lambda-sm-nm`

例如：

- `T400_xB0p030`
- `T450_xB0p025`
- `T500_xB0p040`

每次都重复同一套命令，结果会自动进入不同 workflow 子目录。

这样后续读取、汇总、继续生长、比较不同温度和不同 `xB` 时都不会混乱。
