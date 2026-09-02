# Physical lower-boundary contract

The only resolved lower boundary is the exact binary64 grid edge `Rmin`; it is not a cell centre, occupied bin, or extrapolation point.  A negative growth velocity is outward, a nonnegative velocity has zero external inflow, and boundary diagnostics never update matrix composition.

```json
{
  "GP_release": "not_authorized",
  "J_N_out_positive": true,
  "R_boundary_binary64_hex": "0x1.03b4c5a5fbe5ep-31",
  "R_boundary_m": 4.724027182871601e-10,
  "baseline_R_boundary_binary64_hex": "0x1.03b4c5a5fbe5ep-31",
  "boundary_diagnostics_are_matrix_sources": false,
  "boundary_growth_kernel": "shared_growth_rate_beta",
  "boundary_growth_velocity_expression": "growth_rate_beta(R=Rmin, x_alpha, T, validation_contract)",
  "external_inflow_rule": "zero_when_G_boundary_ge_0_without_subgrid_source",
  "ghost_population": "forbidden",
  "inventory_authority": "Q_total = Q_matrix + Q_beta_resolved",
  "lower_face_location": "Rmin_exact_grid_edge",
  "matrix_closure": "algebraic_from_current_population_only",
  "outflow_rule": "G_boundary_lt_0",
  "outward_radius_direction": "negative_R",
  "particle_inventory_price_radius": "Rmin_exact_grid_edge",
  "radius_domain_m": [
    4.724027182871601e-10,
    1e-07
  ],
  "schema_version": "KWN_PHYSICAL_LOWER_BOUNDARY_CONTRACT_V1",
  "upwind_reconstruction": "piecewise_constant_first_resolved_cell_donor",
  "validation_contract_hash": "d0ff02973ab0f737043e1a40d4f69893a469cbfe2bc4cd22f9e6a410bd0b1333"
}
```
