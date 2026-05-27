# Step 31: Profile-Based Mature GP Initializer Report

This report is populated after running `run_step31_profile_based_mature_gp_init.sh`.

Expected outputs:

- `final_key_summary.csv`
- generated radial profile CSVs from `tools/analysis/generate_observed_gp_profile_bundle.py`
- CUDA run logs and `dynamics_mass_diagnostics.csv` per case

Comparison groups:

- A: hard analytic `eta(r)` + current `local_compensate`
- B: `observed_gp_diffuse`
- C: offline profile bundle (`phi=0`, matched smooth `eta/xB_alpha`)

Key questions:

1. Does the offline profile bundle reduce initial `grad_xB` / estimated `grad_mu`?
2. Does it reduce early `dt*divJ` compared with A/B?
3. Can `eta_peak = 1.0` survive dynamically when initialized with a smooth matched `xB_alpha(r)` profile?
4. Are previous hard-seed failures initialization artifacts rather than proof of thermodynamic impossibility?
