# 246³ A/B/C 6–48 h production launch

## Registered numerical contract

```text
dt_code=0.02
dt_physical_s=0.9909260953431841
final_step=152585
science_steps=0,21798,43596,65393,108989,152585
checkpoint_cadence_steps=3633
checkpoint_cadence_physical_s=3600.034504381788
checkpoint_count_per_path=44
```

Every path starts from its own original hash-pinned 6 h raw fixture. GP,
GP Birth, GP release, external sources, and new beta nucleation are disabled.
The 12 h concentration observation is read-only and cannot stop a conditional
path.

## Frozen identities

```text
source_root=/data/home/luozhiheng/tmp/codex_pf_246cube_three_seed_handoff_v1_20260731
binary_sha256=efb99c707acf7f22899425b8742c7dc06bb3231b9f21604cb5d75cf4565d8c94
parameter_sha256=ecbdd0ac070bdf5e5d214322b5248a08f5ca5dd4e0670427513f5ef977ea977a
profile_library_manifest_sha256=58803a8bc6679b823e45e7a7b85df16ae68efa55338d52d4c4151b414a5ef0fe
replicate_A_fixture_sha256=63a5080b01962bf19f72a37541ffde4302fec3a0e4dc9e519b0f759f30367de6
replicate_B_fixture_sha256=b37682e5aea7cc294a675ce562a34fb0d306181990d40090df1d20779a93980c
replicate_C_fixture_sha256=12c265be4e352392385e689c87ecaea1d18dc364428c111fab5bceb1eac0f981
production_runner_sha256=3d49afc81aa97bf810bc2c5df2d812a8ddd1fa722eb44d417798690a98091c4b
production_auditor_sha256=6ad0de488ed5b06343d99a1953d4006acb8b5fdb7410a765d226ceac5cb7b8bb
ensemble_assembler_sha256=698d6f819acf65e935b638e4974e0a95362492cc44b26ae3c1644309094966fd
```

All three preflights passed, including manifest identity, raw `phi/xB` hashes,
particle/component-list hashes, inventory, grid, elasticity, and forbidden-path
checks.

## Cluster launch

| Path | Slurm job | Output root | Launch state |
|---|---:|---|---|
| A | 73230 | `/data/home/luozhiheng/tmp/pf_246cube_6h48h_A_v1_20260731` | RUNNING on `gpu1`, A100-SXM4-40GB |
| B | 73231 | `/data/home/luozhiheng/tmp/pf_246cube_6h48h_B_v1_20260731` | PENDING, serialized by QOS |
| C | 73232 | `/data/home/luozhiheng/tmp/pf_246cube_6h48h_C_v1_20260731` | PENDING, serialized by QOS |

At launch, A had empty stderr, about 4.4 GiB device memory use, and sustained
99–100% sampled GPU utilization. The `gpu_uvip` QOS currently permits only one
running job for this user, so the three independent jobs are registered but
will execute serially unless the scheduler policy changes.

Final ensemble statistics and scientific grading are not yet available.

## Supplemental redundant launches

At the user's explicit request, two additional non-overwriting copies were
registered without cancelling the original `gpu_uvip` B/C jobs.

### C on `gpu_vip_24h`

```text
job_id=73235
partition=gpu_vip_24h
qos=gpu_vip_24h
output_root=/data/home/luozhiheng/tmp/pf_246cube_6h48h_C_vip24h_v1_20260731
preflight=PASS_246CUBE_6H48H_PRODUCTION_PREFLIGHT_V1
launch_state=PENDING_PRIORITY
```

It uses the same cluster source, `sm_80` binary, parameters, C fixture, and
44-checkpoint contract as the original C job. At registration both A100 GPUs
on `gpu1` were occupied by job 73230 and another user's job 73203, so the
supplemental C copy was queued rather than started.

### B on workstation

```text
driver_pid=154434
main_cuda_initial_pid=154479
device=NVIDIA_GeForce_RTX_5080
output_root=/home/zhiheng/tmp/pf_246cube_6h48h_B_workstation_v1_20260731
runtime_binary_sha256=7581c169fb1d1c16ee60764f418ee9c33508682c8b9dd8cf76b89368c7990602
workstation_runner_sha256=3682feae038c40ff7fe7d071c3aad81966728f8214d5bccedcc760b95d92da47
preflight=PASS_246CUBE_6H48H_PRODUCTION_PREFLIGHT_V1
launch_state=RUNNING
```

The workstation binary is the previously qualified native binary for that
device. Physics source, parameter hash, B fixture hash, endpoints, and
checkpoint cadence are unchanged. Initial measurements were approximately
`1.267 s/step`, GPU utilization 100%, 4680 MiB device memory, and empty
stderr. Its projected pure-compute duration is approximately 53.7 h.
