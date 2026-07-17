# M7 explicit FFT workspace

- Decision: **NOT_RETAINED_ZERO_BENEFIT**
- Source: `main_cuda.cu:31316-31339`
- Old/new operation: optional explicit shared cuFFT work area tested; selected path keeps normal plan ownership
- Source saving at 400^3: **0 GiB**
- Arithmetic/physics/gate change: **none**
- Correctness evidence: mask-123 endpoint bitwise
- Performance evidence: workspace query returned zero bytes on all tested grids
- Rollback/restart disposition: default-off plan teardown tested
- Reason: no memory benefit
