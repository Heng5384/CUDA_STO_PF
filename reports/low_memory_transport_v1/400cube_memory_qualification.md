# 400-cube memory qualification

## Frozen target

- GPU: NVIDIA GeForce RTX 5080
- Reported device memory: 16303 MiB (15.921 GiB)
- Admission ceiling: 85%, or 13.533 GiB
- Frozen path: coarse4 active-manifold IMEX-BDF2 with event rollback buffers,
  adaptive feasible Ctot coordinate, capacity-aware outer face history, and
  outer acceleration OFF.

## Source allocation result

The source-exact persistent `cudaMalloc` ledger gives:

| Configuration | Explicit allocations at 400^3 | Status before cuFFT/context |
|---|---:|---|
| PF only | 19.580 GiB | FAIL |
| PF + elasticity | 34.145 GiB | FAIL |
| PF + elasticity + static GP reservoir structures | 34.145 GiB device-side | FAIL and S3-gated |

Static GP site reservoirs are host vectors in the current source. The third
row therefore has the same device field total as the elastic row; it is not an
integrated runtime claim because the Ctot candidate still gates GP/S3.

The PF-only explicit allocations exceed the 85% ceiling by about 6.05 GiB
before CUDA context, cuFFT plans/work areas, allocator fragmentation, or future
event headroom are counted. Consequently, the frozen implementation cannot be
admitted at 400^3 on this 16 GiB GPU even under the impossible assumption of a
zero-byte cuFFT workspace.

## Evidence boundary

`device_buffer_lifetime.csv` enumerates the persistent source allocation path,
including small solver/reduction packets. `memory_scaling.csv` records the
source totals and the hard source-only rejection. Actual cuFFT work-area sizes,
context loss, observed peak loss, and the largest safe cubic grid remain
pending a clean instrumented build on the target GPU. No 400^3 allocation was
attempted after the source-only gate failed.

Current status: `400CUBE_MEMORY_BLOCKED_PENDING_MEMORY_REFACTOR`.
