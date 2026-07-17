# M6 face streaming

- Decision: **RETAINED**
- Source: `main_cuda.cu:34564-34635; cuda_kernels.cu:6291-6318`
- Old/new operation: three simultaneous positive-face fields -> one x/y/z streamed field with ordered divergence accumulation
- Source saving at 400^3: **0.953674 GiB**
- Arithmetic/physics/gate change: **none**
- Correctness evidence: manufactured periodic incidence test; 1D/32^3/64^3 endpoint bitwise
- Performance evidence: 64^3 wall improves 3.78%; kernel launches increase 100 in 2-step trace
- Rollback/restart disposition: failed-attempt diagnostics materialize temporary faces only on failure
- Reason: saves two full FP64 fields with net measured speed benefit
