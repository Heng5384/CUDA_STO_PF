# CUDA Runtime / Driver Mismatch Fix Report

Date: 2026-06-22

Scope: CUDA runtime/build environment only. No PF physics, nucleation model, CNT, or dynamic-continue code was changed.

## 1. Access Notes

`ssh uvip-cluster` is currently blocked by Tailscale SSH web confirmation:

```text
Tailscale SSH requires an additional check.
```

The same cluster repository was reachable through `cluster-direct`:

```text
/data/home/luozhiheng/CUDA_STO_PF
```

All environment inspection and fixes below were performed against that project path.

## 2. Environment Inspection

### Login / management node

Host:

```text
mgt.gtiit0407
```

Observed:

```text
nvidia-smi: command not found
nvcc: command not found
module list: No Modulefiles Currently Loaded
CUDA_HOME=
LD_LIBRARY_PATH=
Gres=(null) for login/CPU nodes
```

Current `main_cuda` on the login node links to CUDA 12.8 runtime libraries:

```text
libcufft.so.11  => /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcufft.so.11
libcudart.so.12 => /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcudart.so.12
```

Interpretation:

```text
Do not run CUDA binaries directly on the login/management node.
```

That path can produce misleading driver/runtime failures because no GPU driver/runtime environment is active.

### GPU node evidence from historical Slurm logs

Historical GPU jobs under `jobs/logs/` consistently show:

```text
Driver Version: 570.86.10
CUDA Version: 12.8
NVCC=/usr/local/cuda/bin/nvcc
compile command uses /usr/local/cuda or /usr/local/cuda-12.8
arch=sm_80
```

Example historical build line:

```text
/usr/local/cuda/bin/nvcc -arch=sm_80 -O3 -std=c++14 ... -L/usr/local/cuda/lib64 -lcufft -lcudart
```

Available local CUDA directories:

```text
/usr/local/cuda
/usr/local/cuda-12.8
```

Slurm GPU partitions:

```text
gpu_uvip     GRES=gpu:2
gpu_vip_24h  GRES=gpu:2
```

Correct Slurm resource flags used by existing project jobs:

```text
--partition=gpu_uvip
--qos=gpu_uvip
--gres=gpu:1
```

## 3. Version Alignment Table

| Component | Observed / configured | Status |
|---|---:|---|
| GPU driver on historical GPU jobs | 570.86.10 | OK for CUDA 12.8 |
| GPU node CUDA capability reported by nvidia-smi | CUDA Version 12.8 | OK |
| Current binary linked runtime | `/usr/local/cuda-12.8` | OK |
| Historical nvcc | `/usr/local/cuda/bin/nvcc`, resolving to CUDA 12.8 stack | OK |
| Existing `_site_env.sh` default before fix | `cuda/cuda-12.9` | bad default / module missing |
| `_site_env.sh` default after fix | `cuda/cuda-12.8`, `/usr/local/cuda-12.8` preferred | fixed |
| Direct login-node execution | no `nvidia-smi`, no active GPU driver | invalid execution path |

Classification:

```text
GPU node driver/runtime/nvcc: MATCHED
login-node direct CUDA execution: INVALID NODE / fatal for GPU runtime
module default cuda-12.9: CONFIG MISMATCH, fixed to cuda-12.8
```

## 4. Root Cause

The immediate error:

```text
CUDA driver version is insufficient for CUDA runtime version
```

was triggered when `main_cuda` was run directly from the SSH/login environment, not from a Slurm GPU allocation.

Root cause:

```text
CASE D - Slurm GPU node mismatch / login-node execution.
```

Contributing issue:

```text
CASE B - module default mismatch.
```

`jobs/_site_env.sh` defaulted to `cuda/cuda-12.9`, which does not exist on this cluster. Historical jobs fell back to auto-detected `/usr/local/cuda` or `/usr/local/cuda-12.8`, but the warning created an inconsistent and fragile build environment.

Direct answers:

1. Binary compiled with newer CUDA than driver supports?
   - **No for GPU nodes.** Historical GPU node driver is 570.86.10 with CUDA 12.8 support, and binary links CUDA 12.8.
2. Cluster GPU driver outdated?
   - **No evidence.** Historical GPU logs show driver 570.86.10 / CUDA 12.8.
3. Module load CUDA conflict?
   - **Yes.** `_site_env.sh` tried `cuda/cuda-12.9`, a missing module. Fixed to 12.8.
4. Local compile vs cluster runtime mismatch?
   - **Not the main issue.** Current cluster binary links `/usr/local/cuda-12.8`. The observed failure came from running on the wrong node/environment.

## 5. Fix Applied

Changed:

```text
jobs/_site_env.sh
```

Fixes:

1. Default Slurm CUDA module changed from:

```text
cuda/cuda-12.9
```

to:

```text
cuda/cuda-12.8
```

2. CUDA auto-detection candidate order changed to prefer:

```text
/usr/local/cuda-12.8
/usr/local/cuda
/usr/local/cuda-12.9
```

instead of trying 12.9 first.

3. The fixed `_site_env.sh` was synchronized to:

```text
/data/home/luozhiheng/CUDA_STO_PF/jobs/_site_env.sh
```

## 6. Validation Job

Created and submitted:

```text
jobs/generated/test_cuda_runtime_fix.sbatch
```

Submitted Slurm job:

```text
62338
```

Job behavior:

1. allocate one GPU on `gpu_uvip`
2. source fixed `jobs/_site_env.sh`
3. enforce CUDA 12.8:

```text
CUDA_MODULE=cuda/cuda-12.8
CUDA_ROOT=/usr/local/cuda-12.8
CUDA_ARCH=sm_80
```

4. print `nvidia-smi`, `nvcc --version`, module list, `LD_LIBRARY_PATH`
5. run `make clean`
6. rebuild `main_cuda` with CUDA 12.8
7. run a 128³ minimize smoke:

```text
./main_cuda \
  --pf-param-file <generated> \
  --mode=minimize \
  --Nx 128 --Ny 128 --Nz 128 \
  --minimize-max-iter 1 \
  --minimize-dt 0.001 \
  --radius-phys-nm 1.0 \
  --elastic 0
```

8. verify at least one `energy_minimize_*.csv` exists

Current status when this report was written:

```text
PENDING (Resources)
```

Expected logs:

```text
jobs/logs/cuda_runtime_fix_62338.out
jobs/logs/cuda_runtime_fix_62338.err
```

## 7. Test Result

Current result:

```text
Validation job submitted, waiting for GPU resources.
```

The earlier raw-field smoke reached `main_cuda` initialization successfully on the login node and verified the generated raw fields, but failed at CUDA runtime initialization because it was not running inside a GPU allocation.

Final runtime smoke pass/fail must be read from job `62338` after it starts.

## 8. Future Build Recommendation

Use this environment for all GPU builds/runs:

```bash
source jobs/_site_env.sh
export CUDA_MODULE=cuda/cuda-12.8
export CUDA_ROOT=/usr/local/cuda-12.8
export CUDA_ARCH=sm_80
site_require_cuda_env
make clean
make main_cuda "CUDA_ROOT=${CUDA_ROOT}" "CUDA_ARCH=${CUDA_ARCH}" "NVCC=${NVCC}"
```

Run CUDA binaries only under Slurm GPU allocation:

```bash
sbatch -p gpu_uvip --qos=gpu_uvip --gres=gpu:1 <job.sbatch>
```

or for interactive debugging:

```bash
srun -p gpu_uvip --qos=gpu_uvip --gres=gpu:1 --pty bash
```

Do not treat login-node CUDA errors as GPU-node runtime errors.

## 9. Final Status

Root cause fixed at environment-script level:

```text
module/runtime default now targets CUDA 12.8
```

Execution-path fix:

```text
CUDA validation moved from direct SSH/login-node execution to Slurm GPU job
```

Full smoke validation:

```text
pending GPU resources, job 62338
```
