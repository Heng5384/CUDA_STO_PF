# Mechanical boundary-condition audit

The elastic path is a periodic spectral Green-function solver. The k=0 displacement mode is set to zero in both eigenstrain and Green-function kernels. This means the default audited case is a fixed periodic cell / zero imposed homogeneous strain (`E0=0`), not a traction-free variable-cell relaxation. A nonzero uniform external strain is added separately through `E0_*`; no such strain is used here. The solver iterates the heterogeneous stiffness correction (`S_p=C_beta-C_matrix`) up to `elastic_iter_max`.

Consequences for interpretation:

1. `elastic_flat` is a fixed-cell coherent equilibrium unless a separate relaxed-cell implementation is explicitly added.
2. The k=0 handling must not be described as zero macroscopic stress.
3. Finite-particle elastic runs must report fixed-cell stress and elastic energy, with box-size sensitivity.
