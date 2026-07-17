# M3 compact masks

- Decision: **NOT_RETAINED**
- Source: `main_cuda.cu active-mask allocation and consumers`
- Old/new operation: no change; FP64 active-code field retained
- Source saving at 400^3: **0 GiB**
- Arithmetic/physics/gate change: **none**
- Correctness evidence: not implemented
- Performance evidence: not measured
- Rollback/restart disposition: no ABI/reduction migration attempted
- Reason: memory target already met; coordinated ABI/reduction risk not justified
