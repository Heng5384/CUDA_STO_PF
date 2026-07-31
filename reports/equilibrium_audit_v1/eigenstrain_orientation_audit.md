# Eigenstrain and orientation audit

Frozen principal eigenstrain is `[0.046, -0.022, -0.017]`; the supplied rotation is identity, so principal axes are simulation x/y/z. Shear components are zero and are interpreted as tensor shear (not engineering gamma). The chemical isotropic strain is `eps_iso=0.00233`; the runtime writes `eps_iso_over_vB=eps_iso/v_B`. The coherent beta eigenstrain is phase-interpolated with h(phi), and the matrix has zero eigenstrain in the frozen cases.

No orientation averaging or post-hoc rotation is introduced.
