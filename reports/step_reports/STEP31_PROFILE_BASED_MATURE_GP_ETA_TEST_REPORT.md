# Step 31 Priority Test: Profile-Based Mature GP eta-Peak Audit

This report is populated after running `run_step31_profile_based_mature_gp_eta_test.sh`.

Priority question:

Can `eta_peak = 1.0` mature GP be dynamically stable if initialized with a smooth, self-consistent `xB_alpha(r)` profile?

Cases:

- hard compare: `eta_peak = 1.0`, `R = 1 nm`, current `single_sphere + local_compensate`
- offline profile bundles:
  - `eta_peak = 1.0, 0.8, 0.5`
  - `R = 1 nm`
  - `depletion_radius_factor = 5, 8, 12`
  - `xB_background = 0.03`

Metrics:

- initialization mass conservation
- initial `max_abs_grad_xB`
- initial estimated `max_abs_grad_mu`
- `max_abs(dt*divJ)`
- clipping counts
- `eta_max(t)`, `eta_integral(t)`, `V_h(t)`, `R_eff_h(t)`
- survival time / stability classification

Interpretation target:

- If `eta_peak = 1.0` becomes healthy with a matched smooth profile, earlier failures were initialization artifacts.
- If `eta_peak = 1.0` still fails but `0.8/0.5` survive, mature diffuse GP is viable but full-amplitude GP is too aggressive under Scheme A.
- If all fail, Scheme A likely needs additional GP stabilization rather than a better initializer alone.
