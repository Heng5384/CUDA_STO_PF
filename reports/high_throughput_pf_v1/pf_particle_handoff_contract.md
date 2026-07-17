# Conservative PF-to-particle handoff contract

The handoff is offline and default-off. It cannot write `Ctot`, `phi`, GP inventory, or any
runtime field. Its only authority is the accepted PF snapshot.

For every cell, `q_alpha=Ctot-h(phi)*v_B`. The exact snapshot ledger is

```text
M_total = sum(Ctot) + M_GP
        = sum(q_alpha) + sum(h(phi)*v_B) + M_GP.
```

Resolved objects are connected components of a registered `phi` threshold. Diffuse `h(phi)`
inventory is partitioned once among those objects and is never also counted as matrix inventory.
Each particle record includes position, connected-component threshold volume, `h` volume,
equivalent radius, exposed-face area, shape tensor, orientation, local matrix composition,
optional chemical potential, elastic provenance tag, neighboring distance, and GP inventory tag.

Eligibility remains strict: stable particle count and topology over a registered window, no new
nucleation, `R>=2 lambda=8 nm`, threshold-stable identification, relaxed shape, smooth matrix away
from interfaces, PF hard gates passing, and either releasable GP inventory <=5% or an explicit
particle-model GP ledger. This tool does not itself establish eligibility.

Roundtrip acceptance requires relative global mass error <=1e-12, particle beta and matrix
inventory errors <=1e-6, unchanged count, radius error <=0.1%, and no duplicate GP inventory.
The implementation has synthetic two-particle operator tests and a real accepted 32-cube
snapshot roundtrip. The real ledger closes to `2.82816e-13` relative and its particle
partition closes to `3.97557e-13`. Temporal eligibility and PF/oracle overlap remain separate
gates; this contract alone does not authorize a long-time particle model.
