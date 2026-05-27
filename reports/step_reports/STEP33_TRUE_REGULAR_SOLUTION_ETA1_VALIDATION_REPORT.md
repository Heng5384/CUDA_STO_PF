# Step 33 True Regular-Solution Mature GP Validation

## Goal

Evaluate a single mature GP case under:

- `gp_y_update_mode = conservative_y_rhs`
- `gp_init_mode = observed_gp_diffuse`
- `T = 653 K`
- `thermo_convex_extrapolation_enabled = 0`

without modifying solver formulas.

## Case

- `model_mode = gp_zone`
- `xB_background = 0.03`
- `R_GP = 1.0 nm`
- `eta_peak = 1.0`
- `depletion_radius_factor = 2`
- `grid = 96^3`
- `dx = 0.1 nm`
- `dt = 1e-5`
- `steps = 5000`

## Required conclusion

Answer only:

> Under true regular-solution thermodynamics (no convex extrapolation), can the representative mature GP case run to 5000 steps without the `storage_exact`-style early blow-up?

## Outputs

- `final_key_summary.csv`
- `vh_timeseries.csv`
- `run.log`
- `dynamics_mass_diagnostics.csv`
