# Library mapping audit

All 96 particles in each replicate load one exact selected entry from the
frozen elastic target-profile library. Each particle mapping records its
particle/replicate IDs, registered radius, selected entry ID and manifest
hash, center, orientation, source `phi` and
`delta_C_relaxation` hashes, effective \(h\)-volume, and canonical inventory.

Cross-box assembly transports only:

- the complete native 96³ `phi` window; and
- the portable composition perturbation
  \((1-h)(x_B-x_{B,far})\).

Absolute single-particle-box `xB_alpha` is not copied. The target-box matrix
baseline is re-derived from the fixed global inventory. The assembly uses the
bounded union \(1-\prod_j(1-\phi_j)\) in canonical particle-ID order.

Static gates:

- library manifest identity: PASS
- 288/288 particle-library mappings complete: PASS
- exact registered radii only: PASS
- interpolation/scaling/resampling/rotation: not used
- analytic tanh replacement: not used
- multi-particle minimizer/KKT: not invoked
- physical-support and periodic-image overlap: none
