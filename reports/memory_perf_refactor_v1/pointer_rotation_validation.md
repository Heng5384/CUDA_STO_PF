# M2 transaction/history lifetime

- Decision: **RETAINED**
- Source: `main_cuda.cu:30474-30485,36352-36383,38142-38155`
- Old/new operation: seven always-live outer/history fields -> allocate only for paths requiring multiple/accelerated outer iterations
- Source saving at 400^3: **3.33786 GiB**
- Arithmetic/physics/gate change: **none**
- Correctness evidence: method-consistent single-outer path endpoint bitwise
- Performance evidence: D2D volume reduced
- Rollback/restart disposition: accepted state and BDF2 histories remain immutable
- Reason: largest low-risk lifetime removal; this is history elision, not pointer rotation
