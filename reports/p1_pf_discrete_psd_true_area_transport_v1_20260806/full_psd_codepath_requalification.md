# Full-PSD code-path requalification

Status: `DIRECT_DISCRETE_PSD_REFERENCE`

The frozen implementation in `scripts/pf_full_psd_no_dislocation_transport_v1.py` was read and exercised.
`full_psd_precipitate_rate` calls `precipitate_cross_section` with the complete
one-dimensional list of per-object radii and performs `v/V_box * sum_i` along
the particle axis at every frequency.  The long-wavelength branch is
`(4/9)πR²(Δρ/ρ)²(ωR/v)^4`, the geometric branch is `2πR²`, and the code uses
their harmonic interpolation.  Descriptor models are separate functions and
are not used by Model E.

The base rate adds phonon-phonon (`A_N=1.5`), boundary and matrix-point-defect
rates by Matthiessen addition.  The P1 driver adds the frozen V2 `A2*ω²` member
and, only for F/G/H, the frozen interface term.  S11 and S13 are identically
zero; Yu scale 0.1172768 is disabled.  The output is conditional
no-dislocation lattice-style conductivity, not total experimental κ.

Runtime qualification passed scalar-versus-vector direct summation,
monodisperse degeneration, population/volume scaling, permutation, zero
population, and contract guards.  Core SHA-256: `a42f933c43d0adbd4fc9a4d86a7b74bc08d4efe64517cc3fbf1ba177b62dcf66`.
