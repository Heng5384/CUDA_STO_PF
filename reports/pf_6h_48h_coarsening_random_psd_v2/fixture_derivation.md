# V2 random continuous effective population

This fixture replaces the V1 periodic three-radius lattice while keeping the
qualified source and all physical parameters unchanged.

- source commit: `6b69895af2d1b86b99c57c5479ff767349c61efe`
- domain: `246^3`, `dx=1 nm`, `lambda_sm=4 nm`
- temperature: `380 C`
- matrix `xB_alpha=0.006219279767278563`, `xAg=0.0062`
- target `mean(C_B_tot)=0.03`, `beta_fraction=0.02392954476632684`
- effective resolved population: 96 particles

The dynamic critical-radius bracket is independently registered as 8--10 nm
(8 nm clearly dissolving, 10 nm clearly growing), with 9 nm used only as the
working reference. Radii are deterministic truncated-lognormal quantiles,
not integer-grid values and not repeated radius classes. The common material
scale is applied once after field materialization; it is not an independent
normalization of each particle. The realized radii are 8.1678226--10.8744064
nm and the realized CV is 0.0832656 (the truncation and common field scale
narrow the reference CV=0.25).

Centers are generated from the canonical manifest-derived seed with a
periodic hard-core rejection test. The minimum periodic pair distance is
41.2115448 nm and the minimum clearance margin relative to
`Ri+Rj+4 lambda` is 4.8968522 nm. The Cartesian lattice detector rejects the
fixture; no periodic-image overlap or initial connected-component merge is
present.

Profiles use the qualified single-beta profile and the order-independent union
`phi = 1 - product_j(1-phi_j)`. The canonical manifest hash is
`8ce52733e6755b23f4f7d4ba53f0414ef927bb863c3eb4ce9d43cb09d4d5de45`.
No pre-relaxation or post-hoc physical mass projection is used to define the
6 h state.
