# M5 derived-field de-persistence

- Decision: **RETAINED**
- Source: `main_cuda.cu:30443-30447,31647-32070; cuda_common.cu k4 allocation gate`
- Old/new operation: persistent Y rollback, unused divJ spectrum and k4 -> exact reconstruction or no allocation
- Source saving at 400^3: **1.19567 GiB**
- Arithmetic/physics/gate change: **none**
- Correctness evidence: endpoint and event rollback bitwise
- Performance evidence: reconstruction below 5% regression cap
- Rollback/restart disposition: Y rebuilt from authoritative Ctot+phi
- Reason: removes non-authoritative/unused fields
