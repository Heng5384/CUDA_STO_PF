# Frozen V3 threshold contract

Contract: `PHYSICALLY_SCALED_SIGNED_TRANSPORT_DEFECT_GATE_V3`

- `eta_global <= 1e-04`
- `eta_interface <= 1e-03`
- `eta_cell_max <= 1e-03`
- `beta_global <= 1e-05`
- `beta_interface <= 1e-04`
- `beta_interface_excess <= 1e-04`

These thresholds are frozen before the sixth-window holdout. They are the user-pre-registered 10% signed budgets relative to the V2 global and interface magnitude budgets. No observed G9/dt4 value was used to tune them. Every window and the full trajectory must pass.
