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

## 当前工程约定

- `main_cuda` 二进制不纳入 Git 跟踪
- `Results/`、`jobs/logs/*.out`、`jobs/logs/*.err` 已加入忽略规则
- 输出路径逻辑统一收敛到仓库根目录，避免结果散落在脚本目录或提交目录
