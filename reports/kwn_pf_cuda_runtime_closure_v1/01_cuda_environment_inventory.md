# 01 CUDA environment inventory

The validation-only 96³ CUDA run completed on the cluster GPU node recorded below. It used one A100 GPU and no 400³ production work.

```text
2026-09-01T08:10:01Z
gpu1
097a69582ca3747fecd182ed262278885e0fe0c8
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Jan_15_19:20:09_PST_2025
Cuda compilation tools, release 12.8, V12.8.61
Build cuda_12.8.r12.8/compiler.35404655_0
Tue Sep  1 16:10:02 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.86.10              Driver Version: 570.86.10      CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100-SXM4-40GB          Off |   00000000:31:00.0 Off |                    0 |
| N/A   44C    P0             38W /  400W |       1MiB /  40960MiB |     27%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
CUDA_ARCH=sm_80
CUDA_HOME=/usr/local/cuda-12.8
CUDA_ROOT=/usr/local/cuda-12.8
CUDA_VISIBLE_DEVICES=0
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/local/cuda-12.8/lib64
NVCC=/usr/local/cuda-12.8/bin/nvcc
SLURM_CLUSTER_NAME=gtiit0407
SLURM_CONF=/etc/slurm/slurm.conf
SLURM_CPUS_ON_NODE=4
SLURM_CPUS_PER_TASK=4
SLURM_GPUS_ON_NODE=1
SLURM_GTIDS=0
SLURM_JOBID=89533
SLURM_JOB_ACCOUNT=wngroup
SLURM_JOB_CPUS_PER_NODE=4
SLURM_JOB_END_TIME=1788293400
SLURM_JOB_GID=1001
SLURM_JOB_GPUS=0
SLURM_JOB_ID=89533
SLURM_JOB_NAME=kwn_pf_cuda_ae_v1
SLURM_JOB_NODELIST=gpu1
SLURM_JOB_NUM_NODES=1
SLURM_JOB_PARTITION=gpu_uvip
SLURM_JOB_QOS=gpu_uvip
SLURM_JOB_START_TIME=1788250200
SLURM_JOB_UID=1005
SLURM_JOB_USER=luozhiheng
SLURM_LOCALID=0
SLURM_NNODES=1
SLURM_NODEID=0
SLURM_NODELIST=gpu1
SLURM_NPROCS=1
SLURM_NTASKS=1
SLURM_OOM_KILL_STEP=0
SLURM_PRIO_PROCESS=0
SLURM_PROCID=0
SLURM_SUBMIT_DIR=/data/home/luozhiheng/tmp/kwn_pf_cuda_runtime_closure_v1_097a69582ca3_20260901T080254Z/source
SLURM_SUBMIT_HOST=login1
SLURM_TASKS_PER_NODE=1
SLURM_TASK_PID=3892589
SLURM_TOPOLOGY_ADDR=gpu1
SLURM_TOPOLOGY_ADDR_PATTERN=node
SLURM_TRES_PER_TASK=cpu=4
```
