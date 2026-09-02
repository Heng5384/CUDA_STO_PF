# Canonical smooth initial measure

Status: `PASS_CANONICAL_INITIAL_MEASURE_IDENTITY`.  Eulerian reads the canonical 3200-cell number vector directly; cohort nodes are deterministic positive Gauss--Legendre quadrature from those same cells.

The authority is four points per cell.  Its M0/M3, beta inventory and total inventory errors are all at or below 1e-12.  The independent subcell CDF error falls from `2.384306e-03` (1 point/cell) to `3.632143e-04` (4 points/cell); refinement is `True`.

The reported lower-tail quadrature diagnostic is `1.814518e-04` and is evaluated against the exact piecewise-cell measure, not bin centres.  Fixed cohort IDs, node radii, weights and canonical-cell indices are archived in `canonical_cohort_quadrature_v1.npz`.
