# M4 scratch alias

- Decision: **RETAINED**
- Source: `main_cuda.cu:30898-30905`
- Old/new operation: separate d_Y_k spectrum -> alias d_phi_k after non-overlapping lifetime proof
- Source saving at 400^3: **0.479221 GiB**
- Arithmetic/physics/gate change: **none**
- Correctness evidence: 100-step endpoint bitwise
- Performance evidence: no measurable regression
- Rollback/restart disposition: single owner prevents double free
- Reason: one full complex spectrum removed
