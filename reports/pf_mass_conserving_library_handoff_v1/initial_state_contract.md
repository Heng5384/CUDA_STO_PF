# Mass-conserving library handoff V1: initial-state contract

## Formal class

```text
MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1
```

This class means only:

1. every resolved beta particle is mapped to one exact, hash-pinned entry in
   the frozen elastic target-profile library; and
2. the assembled periodic field closes the registered global canonical
   Ag/B inventory.

It is a **mass-conserving, profile-library-assembled conditional 6 h handoff
state**.  It is not a common multi-particle chemical/elastic equilibrium and
is not an experimentally unique 6 h microstructure.

## Frozen physical and numerical boundary

- temperature: 380 °C;
- grid: 96³, `dx=1 nm`;
- diffuse-interface width: `lambda_sm=4 nm`;
- fixed periodic cell and identity/[100] eigenstrain variant;
- registered radii only: 8.0–11.5 nm in 0.5 nm increments;
- no radial interpolation, scaling, rotation, analytic tanh replacement,
  clipping, normalization, or pre-relaxation;
- GP, GP Birth, GP release, external source, and new beta nucleation: OFF;
- Ji-Chen conserved-Y zero mode:
  `PF_CONSERVED_Y_ZERO_MODE_V1/HOST_NEWTON_BISECTION_V1`;
- time-level/reaction provenance:
  `SM_EXPLICIT_CONTEXT_N_V1/SM_TANGENT_N_V1`.

## Assembly

The beta field uses the bounded, order-independent union

\[
\phi_{\rm total}=1-\prod_j(1-\phi_j).
\]

Cross-box composition transport uses each frozen entry's
`delta_C_relaxation=(1-h_j)(x_{B,j}-x_{B,\mathrm{far},j})`.  Its matrix
deviation is reconstructed algebraically and stored in the target phase as

\[
\delta C_{\rm target}
=(1-h_{\rm total})\sum_j
\frac{\delta C_j}{1-h_j}.
\]

The target-box matrix baseline is solved once from the global inventory.
Absolute isolated-box `xB_alpha` fields are not copied into the target box.

## Time semantics

No multi-particle constrained minimizer is run.  At `t=6 h` the complete
periodic elastic field is recomputed and ordinary PF dynamics starts
immediately.  Particle profiles are not locked.  Restructuring, dissolution,
growth, merging, and coarsening after this point are physical evolution and
their time may not be discarded or relabelled.

## Implemented schema/provenance

The fixture records all required policy, library mapping, target inventory,
derived baseline, canonical ledger, zero-mode provenance, and time-semantic
fields.  Checkpoint V3 additionally pins:

```text
initial_state_class
fixture_manifest_sha256
profile_library_manifest_sha256
```

Historical V2 checkpoints remain readable only with the explicit
`LEGACY_UNSPECIFIED` initial-state identity.

