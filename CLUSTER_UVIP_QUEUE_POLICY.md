# Cluster gpu_uvip Queue Occupancy Policy

This file is a project-level operating rule for CUDA_STO_PF cluster runs.
It is intentionally strict because gpu_uvip queue position affects whether
long simulation batches can start promptly.

## Iron Rule

Whenever the user asks to run a workload on the cluster gpu_uvip queue:

1. Submit the current target workload first.
2. Submit at least one tail placeholder, continuation, or resume job behind the target workload.
3. Verify the target job and the tail job exist in `squeue`.
4. Only after verification, move or cancel older jobs that are blocking the target workload.
5. Keep a placeholder or resume job at the back of the active project queue whenever possible.

This is the default project behavior. Do not treat it as optional.

## Never Hold Jobs

Do not use:

```bash
scontrol hold <job_id>
```

Holding a job can let other users' jobs move ahead of the project queue.
Use dependencies, continuation jobs, or same-root resume jobs instead.

## Safe Reprioritization Order

Required order for cluster reprioritization:

1. Prepare the target workload scripts.
2. Submit the target workload to `gpu_uvip`.
3. Submit one or more tail continuation jobs, for example with `--dependency=afterany:<target_job_id>`.
4. If there are older project jobs that must resume later, submit same-root backup or resume jobs behind the final tail job.
5. Verify every replacement, continuation, and dependency with `squeue` and, when needed, `scontrol show job`.
6. Only then cancel the older blocking jobs.

If any replacement or placeholder submission fails, do not cancel the blocking job.

## Preferred Mechanisms

Use these mechanisms in preference order:

1. Slurm dependency chain, such as `afterany:<job_id>`.
2. Same-root resume-or-skip job for long sweeps.
3. Serial resume wrapper when array submission is blocked by QOS limits.
4. Minimal tail placeholder script that exits safely or runs a resume check.

Do not fork replacement workflow roots unless the original output tree is
corrupted or the user explicitly requests a clean rerun.

## Old Jobs Behind New Workloads

When a new urgent gpu_uvip workload must start before older project jobs:

1. Submit the new workload.
2. Submit a tail continuation for the new workload.
3. Put older pending jobs behind the final tail job using dependencies when possible.
4. For older running jobs, submit verified same-root backup jobs behind the final tail job before canceling the running copy.

This preserves the project's queue occupancy while preventing other users'
jobs from taking the released slot before the new target workload.

## Required Report Contents

Every queue manipulation report should record:

- target job IDs
- tail placeholder or continuation job IDs
- old jobs moved behind the target workload
- old jobs canceled
- dependency chain used
- verification command output or a summary of it
- whether any job was left without a verified replacement

## Current Project Memory

For CUDA_STO_PF, the standing rule is:

> If the user asks to run on the cluster, use the gpu_uvip queue occupancy
> mechanism. Always keep a placeholder, continuation, or resume script behind
> the current target workload. Submit and verify the target and placeholder
> before canceling any existing job. Never use `hold`.
