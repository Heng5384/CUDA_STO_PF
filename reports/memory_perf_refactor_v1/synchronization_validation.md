# M8 copy/synchronization

- Decision: **RETAINED**
- Source: `main_cuda.cu:38387-38467`
- Old/new operation: eight full-field non-authoritative diagnostic copies every step -> CSV cadence/end only
- Source saving at 400^3: **0 GiB**
- Arithmetic/physics/gate change: **none**
- Correctness evidence: hard gates and failure diagnostics remain per-step
- Performance evidence: 2-step D2H trace reduced 42.16%
- Rollback/restart disposition: decision-boundary checks unchanged
- Reason: removes observer traffic without delaying authoritative failure checks
