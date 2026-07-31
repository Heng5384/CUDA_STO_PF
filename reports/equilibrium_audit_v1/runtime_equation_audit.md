# Runtime equation audit

## Chemical thermodynamics

`thermo_utils.h` implements SGTE Pb/Ag/Te references, PbTe and Ag2Te standard energies, and a regular pseudo-binary interaction `L(T)=41212.9-18.05T` J/mol. The raw chemical potentials are

- `mu_A = G0_PbTe + RT ln(1-xB) + L xB^2`;
- `mu_B = G0_Ag2Te + RT ln(xB) + L (1-xB)^2`.

The optional convex extrapolation is runtime-disabled in the frozen audit (`thermo_convex_extrapolation_enabled=0`); the cutoff `X_LIMIT_CONVEX=0.09` is therefore not used.

## PF variational terms

`h(phi)=phi^3(6 phi^2-15 phi+10)` and `g(phi)=phi^2(1-phi)^2`. The phase RHS contains chemical reaction drive, double-well, gradient term and, when enabled, `-sigma: d(eps0)/dphi + 0.5 h'(phi) Q`. The conserved composition uses the logit `Y` representation and the runtime's legacy conserved update; no mass projection is allowed in this audit.

## Composition and units

`xB` is the pseudo-binary Ag2Te fraction (beta endpoint xB=1). For an Ag atomic fraction reported in the Ag sublattice convention, the project converter uses `xAg=2 xB/(2+xB)` and inverse `xB=2 xAg/(2-xAg)`. `lambda_sm=4 nm`, `pf_dx=1 nm`, so the interface resolution is 4 cells and `L_ref=5 lambda_sm`; physical time is `t_phys=t_code*(L_ref^2/D_alpha_phys)`.

## GP/nucleation gates

The production source contains GP, birth, release and beta nucleation paths, but all are explicitly OFF for this audit. No source, clipping, retuning or physical mass projection is permitted.

## Workstation startup verification

The isolated GP-OFF fixture printed `xB_eq=4.66495182e-03`, `lambda_sm/dx=4`, `t_real_unit_s=4.95463048e+01`, and `gp_paths_enabled=false`. This is the runtime check of the static equation audit, not a fit to an experimental concentration.

The authoritative rerun uses `codex/pf-zero-mode-restart-provenance-v1` at commit `6b69895af2d1b86b99c57c5479ff767349c61efe`; its startup printed the same `xB_eq`, `dt=1e-4`, `lambda_sm/dx=4`, and `gp_paths_enabled=false`.
