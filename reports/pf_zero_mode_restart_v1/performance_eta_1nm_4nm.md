# PF zero-mode performance: eta 1 nm / 4 nm contract

## Decision

```text
performance_status=PASS_PF_ZERO_MODE_ETA_1NM_4NM_PERFORMANCE_MATRIX_V1
paired_persistent_reduction=true
per_iteration_cuda_allocation=false
per_iteration_full_field_D2H=false
```

Cluster target `72653` and dependency tail `72654` completed successfully.
Output root:

```text
/data/home/luozhiheng/tmp/pf_zero_mode_restart_eta1nm4nm_perf_q12_20260728
```

Each case ran 192 steps at T400 with `dt_code=0.02`, or
0.8225917084910954 s/step and 157.9376080302903 s total physical time. The
baseline uses the legacy k=0 handling; the selected run uses
`PF_CONSERVED_Y_ZERO_MODE_V1`. No checkpoints were written in this timing
matrix.

| Grid | Legacy wall s/step | Zero-mode wall s/step | Wall overhead | Zero-mode final mean mass error |
|---:|---:|---:|---:|---:|
| 64³ | 0.001842 | 0.002019 | 9.61% | 0 |
| 128³ | 0.010314 | 0.010706 | 3.80% | \(3.4694\times10^{-18}\) |
| 195³ | 0.048829 | 0.050361 | 3.14% | \(3.9251\times10^{-18}\) |

The paired reduction evaluates total mass and its derivative in one
deterministic GPU reduction tree, reuses caller-owned scratch memory, and
copies only the final two FP64 scalars to the host for Newton/bisection. This
removes the previous per-evaluation allocations and duplicate reduction
synchronizations.

The 195³ result is the most representative result in this matrix: the
zero-mode conservation mechanism adds 1.532 ms/step, or 3.14%. The overhead
falls with grid size because the base FFT/field work grows faster than the
small scalar transaction.

The 64³ continuous/restart qualification reported 21.84% when compared with a
checkpoint-free baseline, but that number includes two complete checkpoint
writes in the selected run and therefore is not the production solver
overhead. The checkpoint-free timing matrix above is the accepted performance
comparison.

## Large-grid qualification

Cluster target `72659` and dependency tail `72660` completed successfully.
Both grids ran 64 steps, corresponding to 52.64586934343011 s physical time.
The benchmark-only VTK gate was enabled so full-field output did not distort
the result or leave multi-gigabyte artifacts.

```text
status=PASS_PF_ZERO_MODE_ETA_1NM_4NM_LARGE_GRID_PERFORMANCE_V1
output_root=/data/home/luozhiheng/tmp/pf_zero_mode_restart_eta1nm4nm_large_q13_20260728
unexpected_case_stderr_lines=0
driver_fatal_count=0
vtk_file_count=0
```

| Grid | Legacy wall s/step | Zero-mode wall s/step | Wall overhead | GPU memory | Final mean mass error |
|---:|---:|---:|---:|---:|---:|
| 400³ | 0.164405 | **0.176848** | 7.57% | 10.96 GB | \(-3.6380\times10^{-18}\) |
| 512³ | 0.312475 | **0.340115** | 8.85% | 22.45 GB | \(-6.9389\times10^{-18}\) |

At the selected physical timestep, the checkpoint-free estimates are:

| Grid | Steps/s | Simulated physical time per wall second | Wall time for 1 physical hour | Wall time for 6 physical hours |
|---:|---:|---:|---:|---:|
| 400³ | 5.65 | 4.65 s/s | 12.90 min | 77.40 min |
| 512³ | 2.94 | 2.42 s/s | 24.81 min | 148.85 min |

The zero-mode transaction itself measured 28.52 ms/step on 400³ and
55.28 ms/step on 512³, approximately 16.1% and 16.3% of the selected-path
wall time. The end-to-end difference from the legacy comparator is smaller
(7.57% and 8.85%). Thus the main cost remains full-grid FFT/field evolution;
the host scalar solve is a secondary, bounded optimization target rather than
a blocking CPU bottleneck.

Large-grid source identity:

```text
54e6a37f5b867891261001087e38183fa358056a42353961713538d944ab1724  main_cuda
e25de4dc23a5cbf961b0594ae40e466727bfe43c6118d24ce0119182f7457237  main_cuda.cu
f27db426148e25d6d9b221ad24e5013c522d8108c08f86db036807471faee0c6  cuda_kernels.cu
332449e5769845a336acd87fd9a800fd557b41f81a8a002e416d83fef52ca818  io_vtk_cuda.h
559eecf734041782c9fc9e36406358c75ad5e4ae2760d174d9cf10616840e453  pf_zero_mode_checkpoint.cpp
9d91278bd9b333a41a6d418b4ff3b46eefb2c420875cad9671d0864296e35008  pf_zero_mode_checkpoint.h
```
