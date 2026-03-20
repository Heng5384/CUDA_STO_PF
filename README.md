# CUDA_STO_PF

基于 CUDA 的相场模拟项目，主程序入口为 `main_cuda.cu`，结果统一输出到仓库根目录下的 `Results/`。

## 项目结构

- `main_cuda.cu`: 主程序入口，负责参数解析、初始化、时间推进、最小化流程和输出
- `cuda_kernels.cu`: 主要 CUDA kernel 与 GPU 归约/诊断实现
- `cuda_common.cu`, `cuda_common.h`: 公共 CUDA 工具、k 空间构建与 scratch arena
- `pf_params.h`: 参数结构定义
- `jobs/`: 本地运行脚本与 Slurm 提交脚本
- `Results/`: 模拟结果输出目录（已被 Git 忽略）

## 构建

默认配置面向当前工作站环境：

- `CUDA_ROOT=/usr/local/cuda-12.9`
- `CUDA_ARCH=sm_120`

直接编译：

```bash
make main_cuda
```

如需覆盖工具链或架构：

```bash
make main_cuda CUDA_ROOT=/usr/local/cuda-12.9 CUDA_ARCH=sm_120 NVCC=/usr/local/cuda-12.9/bin/nvcc
```

## 运行

本地 dynamics 入口：

```bash
./jobs/run_dynamics_local.sh 64 64 64 0.01 100 50 10 1
```

常见 positional 参数顺序：

```text
Nx Ny Nz dt nsteps out_every csv_out_every elastic_enabled
```

最小化模式可直接调用主程序，例如：

```bash
./main_cuda 64 64 64 0.05 500 50 10 1 \
  --mode=minimize \
  --minimize-full-model \
  --minimize-max-iter 500 \
  --minimize-dt 0.05 \
  --V0 0.1
```

也可以先用物理参数生成一份 PF 覆盖文件，再喂给 `main_cuda`：

```bash
python3 Unit_Psedobinary.py \
  --input-json physical_inputs.example.json \
  --output-json /tmp/generated_pf.json \
  --output-pf-param-file /tmp/generated_pf.params \
  --print-main-cuda-cmd

./main_cuda 64 64 64 0.05 500 50 10 1 \
  --pf-param-file /tmp/generated_pf.params \
  --mode=minimize \
  --minimize-full-model \
  --minimize-max-iter 500 \
  --minimize-dt 0.05 \
  --V0 0.1
```

## 真实物理参数到 PF 输入

`Unit_Psedobinary.py` 现在可以把真实物理输入自动换算成 `main_cuda` 可直接读取的 `key=value` 参数文件。

现在的约定是：

- `main_cuda` 不再依赖内置默认物理参数启动
- `--pf-param-file` 提供的是完整物理输入，不是对旧默认值的“覆盖”
- 如果缺少 `--pf-param-file`，或者参数文件缺少关键物理量，程序会直接报错退出

常用流程：

```bash
python3 Unit_Psedobinary.py --write-example-json physical_inputs.example.json
python3 Unit_Psedobinary.py \
  --input-json physical_inputs.example.json \
  --output-json /tmp/generated_pf.json \
  --output-pf-param-file /tmp/generated_pf.params
```

生成结果说明：

- `/tmp/generated_pf.json`：完整换算结果、尺度信息和元数据
- `/tmp/generated_pf.params`：`./main_cuda --pf-param-file` 可直接读取的参数覆盖文件

当前已自动覆盖的核心字段包括：

- 空间离散：`dx`、`dy`、`dz`
- 热力学与相场：`temperature_C`、`mu_reference_scale`、`kappa_phi`、`L_phi`
- 扩散与体积：`D_alpha`、`D_compound`、`Vm_compound`、`dVm_alpha_dxB`
- 弹性：`S_*`、`S_p_*`、`elastic_shift_dimless`、`eps_iso_over_vB`
- 本征应变：`eps_xx00`、`eps_yy00`、`eps_zz00`、`eps_yz00`、`eps_xz00`、`eps_xy00`

弹性矩阵的约定是：

- `C_tensor_PbTe_GPa`：输入基体完整刚度矩阵，生成 `S_*`
- `C_tensor_Ag2Te_GPa`：输入析出相完整刚度矩阵
- 写入 `main_cuda` 的不是完整析出相刚度，而是差值 `S_p_* = C_Ag2Te - C_PbTe`

这里的 JSON 输入 `dx` 是真实物理网格间距，单位是米。脚本会自动把它换成 `main_cuda` 当前使用的长度单位（按 `nm` 约定写入 `dx/dy/dz`）：

- `dx = 1.0e-9` 表示 `1.0 nm`，生成后 `main_cuda` 里的 `dx=1.0`
- `dx = 1.0e-10` 表示 `0.1 nm`，生成后 `main_cuda` 里的 `dx=0.1`

各向同性化学膨胀这项现在建议直接填：

- `eps_iso`：无量纲各向同性化学膨胀应变

脚本会自动换算为：

- `eps_iso_over_vB = eps_iso / v_B`

当前示例默认给的是：

- `eps_iso = 0.00233`

`main_cuda` 启动时会在开头打印：

- `dx / dy / dz`
- 单元物理尺寸 `cell_size_phys_nm`
- 系统物理尺寸 `system_size_phys_nm`
- `lambda_sm/dx`

目前已接入的自动检查：

- 当 `lambda_sm/dx <= 4` 时，`Unit_Psedobinary.py` 会直接给出 `界面分辨率不足` 警告
- `main_cuda` 启动时也会再次提示同一条警告，避免误跑

### 本征应变输入方式

`Unit_Psedobinary.py` 支持两种本征应变物理输入方式，二选一即可：

1. 直接给模拟坐标系下的对称小应变张量 `eigenstrain_tensor`
2. 给主应变 `eigenstrain_principal`，再配一个从主轴到模拟坐标系的旋转矩阵 `eigenstrain_rotation_matrix`

其中：

- 如果 `eigenstrain_rotation_matrix` 是单位矩阵
  `[[1,0,0],[0,1,0],[0,0,1]]`
  就表示没有旋转
- 也就是主应变方向和模拟坐标系完全一致

示例 1：直接填 3x3 张量

```json
"eigenstrain_tensor": [
  [0.046, 0.000, 0.000],
  [0.000, -0.022, 0.000],
  [0.000, 0.000, -0.017]
]
```

示例 2：填主应变和旋转矩阵

```json
"eigenstrain_principal": [0.046, -0.022, -0.017],
"eigenstrain_rotation_matrix": [
  [1.0, 0.0, 0.0],
  [0.0, 1.0, 0.0],
  [0.0, 0.0, 1.0]
]
```

约定说明：

- 剪切分量使用张量应变定义，不是工程剪切应变 `gamma_ij`
- 如果同时提供 `eigenstrain_tensor` 和 `eigenstrain_principal`，脚本优先使用 `eigenstrain_tensor`
- 示例文件 [physical_inputs.example.json](/Users/heng/Documents/GitHub/CUDA_STO_PF/physical_inputs.example.json) 默认给的是主应变加单位旋转
- 示例文件中的 `_units` 和 `_notes` 只是说明字段，脚本会自动忽略，不影响直接读取
- 生成后的 `eps_xx00..eps_xy00` 会自动写入 `.params` 文件，无需再手抄到命令行

## 输出约定

- 所有模拟结果写入 `Results/`
- dynamics 模式会输出 `phi/xB/xBtot`，可选输出诊断场
- minimize 模式按 case 建子目录，并额外输出 `final phi`
- `jobs/logs/` 用于脚本日志与 Slurm 日志

## Jobs 脚本

`jobs/README.md` 记录了当前脚本分工。提交 Slurm 作业时，建议从仓库根目录执行，例如：

```bash
sbatch jobs/submit_minimize_init_cases.sbatch
```

### jobs/sh 与 sbatch 的物理参数覆盖

`jobs/` 里的本地脚本和 `sbatch` 脚本现在都会先走这条链路：

```text
physical_inputs.example.json
  -> shell/sbatch 环境变量覆盖物理参数
  -> jobs/generated/physical_inputs_<tag>.json
  -> Unit_Psedobinary.py
  -> jobs/generated/pf_params_<tag>.params
  -> ./main_cuda --pf-param-file ...
```

也就是说：

- `physical_inputs.example.json` 是基准物理参数
- `sh` / `sbatch` 里设置的物理环境变量会覆盖这个基准文件
- 每个 case 都会生成自己对应的 `.params`
- `main_cuda` 实际读取的是生成后的 `.params`，不是直接读基准 JSON

目前常用的物理覆盖环境变量包括：

- `TEMP_C`：覆盖 `temperature_C`
- `DX_M`：覆盖真实网格间距 `dx`，单位 `m`
- `GAMMA_JM2`：覆盖 `gamma`
- `LAMBDA_SM_M`：覆盖 `lambda_sm`
- `VF_INIT` / `VF_TARGET`：覆盖 `vf_init` / `vf_target`
- `EPS_ISO`：覆盖 `eps_iso`

如果要一次覆盖多项，也可以直接传：

- `PHYSICAL_OVERRIDE_JSON`：一个 JSON 对象字符串
- `PHYSICAL_OVERRIDE_FILE`：一个 JSON 文件路径

示例 1：在 Slurm 提交时覆盖温度和体积分数

```bash
TEMP_C=410 VF_TARGET=0.12 sbatch jobs/submit_minimize_init_cases.sbatch
```

示例 2：在本地运行时覆盖温度和真实网格间距

```bash
TEMP_C=410 DX_M=1e-10 ./jobs/run_dynamics_local.sh --no-build 64 64 64 0.01 10 10 1 1
```

需要注意的是，像下面这些量仍然是运行控制参数，不属于物理输入 JSON 的换算范围：

- `RADIUS`
- `XB_OUT`
- `NSTEPS`
- `MIN_DT`
- `OUT_EVERY`

它们现在仍然通过 `main_cuda` 命令行直接控制。

## 当前工程约定

- `main_cuda` 二进制不纳入 Git 跟踪
- `Results/`、`jobs/logs/*.out`、`jobs/logs/*.err` 已加入忽略规则
- 输出路径逻辑统一收敛到仓库根目录，避免结果散落在脚本目录或提交目录
