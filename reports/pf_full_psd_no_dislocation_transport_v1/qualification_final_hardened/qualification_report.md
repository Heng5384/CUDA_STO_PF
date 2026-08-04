# PF full-PSD no-dislocation transport V1 qualification

Final status: `PASS_PF_FULL_PSD_NO_DISLOCATION_TRANSPORT_INTERFACE_V1`

This qualifies a conditional resolved-PF-PSD transport interface. It does not claim absolute experimental lattice-thermal-conductivity reproduction.

| Gate | Status |
|---|---|
| frozen_contract | PASS |
| monodisperse_degeneracy | PASS |
| direct_particle_sum | PASS |
| psd_bin_refinement | PASS |
| moment_reconstruction | PASS |
| debye_integration | PASS |
| historical_6h48h_interface_smoke | PASS |
| determinism | PASS |
| abc_authority_selector | PASS |

Frozen physics: `A_N=1.5`, `S11=0`, `S13=0`, no Yu refit scale.

The historical 6--48 h trajectory is used only as an interface smoke. Its overall PSD observations are admissible; no post-merge independent-particle lineage claim is made.

A/B/C ensemble production remains pending until exactly one complete PASS authority is selected for each experiment-matrix-anchored replicate.
