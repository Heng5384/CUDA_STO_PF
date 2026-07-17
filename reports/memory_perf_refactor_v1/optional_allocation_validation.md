# M1 optional allocation

- Decision: **RETAINED**
- Source: `main_cuda.cu:30461-30468,31632-31818`
- Old/new operation: three persistent event snapshots -> two zero-increment aliases plus deterministic Y reconstruction
- Source saving at 400^3: **1.43051 GiB**
- Arithmetic/physics/gate change: **none**
- Correctness evidence: 100-step endpoint and forced event rollback bitwise
- Performance evidence: no easy-path regression
- Rollback/restart disposition: event macro ownership and free path explicitly versioned
- Reason: removes event-only full fields without weakening event safety
