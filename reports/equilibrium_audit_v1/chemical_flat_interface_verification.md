# Chemical flat interface verification

The analytic root and conversion are frozen before the slab run. A first attempt to use the legacy `oneD_test_mode` key was ignored by the current parser (the startup remained the standard 3-D seed), so it is not counted as a slab result.

The raw-field slab path was also tested. The frozen zero-mode runtime fails
closed when raw/VTK continuation is combined with
`PF_CONSERVED_Y_ZERO_MODE_V1` (`zero-mode restart cannot be mixed with VTK/raw-field continuation`). The corresponding no-zero-mode raw slab drifted by
approximately `1.5e-5` over its short run, exceeding the `1e-8` hard mass
criterion.

For method diagnosis only, the isolated fresh-raw-profile extension was used
to run a two-interface `128^3` slab at five compositions.  Its wide bracket
had negative `Delta_vf` at `xAg=0.003` and positive `Delta_vf` at `xAg=0.008`,
while the analytic root `xAg=0.004654096` had near-zero short-time drift.
This is qualitative isolated-extension evidence, not a clean-branch numerical
qualification.  Full values and hashes are in
`chemical_flat_interface_extension_diagnostic.md`.

Route B is therefore still blocked on the clean branch, but the isolated
extension demonstrates that the sign-bracket logic is implementable without
changing thermodynamics or the conservation target.
