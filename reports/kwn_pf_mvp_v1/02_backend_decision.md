# KWN backend decision

Decision: `INTERNAL_KWN_BACKEND_SELECTED`.

## Evidence

No Kawin installation, importable module, pinned package version, environment
lockfile, or repository integration was found in the isolated worktree.  A
Kawin backend therefore cannot supply a reproducible result for this MVP.

The selected backend is `internal_kwn_finite_volume_v1`, implemented under
`src/kwn_mvp/`.  It is a two-population (g and beta) radius-space
finite-volume solver with:

- logarithmic SI radius bins;
- conservative upwind transport, adaptive size-CFL limited to `<= 0.4`, and
  exact accounting of lower-boundary dissolution into the matrix ledger;
- fail-closed Rmax overflow detection rather than particle deletion;
- separate GP and beta molar-volume/composition definitions;
- restart state plus configuration provenance;
- `off`, prescribed-source, and explicitly labelled effective-CNT GP modes;
- no GP-to-beta conversion term and no beta CNT/birth mode.

The equilibrium law in the generic solver is

```text
x_eq(R) = x_eq,infinity * exp(((2*gamma/R + E_el) * V_m)/(R_gas*T)).
```

Its `approximate_dilute` thermodynamic adapter is explicitly labelled
`APPROXIMATE_BETA_THERMO_FOR_EFFECTIVE_MVP`; it is not a silent substitute for
the blocked PF contract.

## Required qualification

Numerical gates N1–N7 cover zero-mobility invariance, dissolution, growth,
closed coarsening, radius-bin convergence, timestep convergence, and
checkpoint/restart.  N8 verifies the required Ag-at.% observation mapping
roundtrip across the supplied AQ/6 h/48 h range.  A separate all-state ledger
invariant checks the maximum relative inventory residual against `1e-10`.
These checks qualify the KWN implementation itself; passing them does not
resolve PF readiness.

## Fallback limitation

An internal KWN result cannot establish PF kinetics, spatial morphology,
elastic response, or a physical GP nucleation mechanism.  In particular, it
cannot repair the unresolved PF thermodynamic contract, manufacture the
missing PF subgrid-beta state, or convert the legacy eta/GP surrogate into a
validated KWN phase.  Those limitations are reported rather than hidden.
