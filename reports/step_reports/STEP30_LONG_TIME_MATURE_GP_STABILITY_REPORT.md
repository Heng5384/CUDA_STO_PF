# Step 30 Long-Time Mature GP Stability Study

This report is the Step 30 entry point for long-time GP-only observed-GP diagnostics.

Status: cluster submission prepared.

## Goal

Determine whether `observed_gp_diffuse` mature GP zones converge to:

- stable finite GP
- very slow dissolution
- slow coarsening/growth

without changing:

- `storage_exact`
- transport PDE
- elastic solver
- event logic
- stochastic insertion
- feasibility gate
- throttling
- `two_phase`
- GP thermodynamics

## Cluster run layout

Primary/secondary cases submitted through:

- `/Users/heng/Documents/GitHub/CUDA_STO_PF/jobs/submit_step30_long_time_mature_gp_stability.sbatch`

Post-run aggregation through:

- `/Users/heng/Documents/GitHub/CUDA_STO_PF/jobs/submit_step30_long_time_mature_gp_stability_aggregate.sbatch`

Driver script:

- `/Users/heng/Documents/GitHub/CUDA_STO_PF/run_step30_long_time_mature_gp_stability.sh`

Remote output root:

- `/tmp/step30_long_time_mature_gp_stability`

Expected generated outputs after completion:

- `/tmp/step30_long_time_mature_gp_stability/final_key_summary.csv`
- `/tmp/step30_long_time_mature_gp_stability/reports/step_reports/STEP30_LONG_TIME_MATURE_GP_STABILITY_REPORT.md`

## Cases

Array cases:

1. `xB_alpha = 0.03`, `R_target = 1 nm`, `eta_peak = 0.2`, `depletion_radius_factor = 5`
2. `xB_alpha = 0.03`, `R_target = 1 nm`, `eta_peak = 0.3`, `depletion_radius_factor = 5`
3. `xB_alpha = 0.05`, `R_target = 1 nm`, `eta_peak = 0.2`, `depletion_radius_factor = 5`
4. `xB_alpha = 0.05`, `R_target = 1 nm`, `eta_peak = 0.3`, `depletion_radius_factor = 5`

All runs use:

- `gp_init_mode = observed_gp_diffuse`
- `gp_nuc_enabled = 0`
- `gp_to_beta_enabled = 0`
- `y_update_mass_projection_enabled = 0`
- `dt = 1e-5`
- `steps = 50000`
- `grid = 96^3`

## Planned interpretation

The final classification will distinguish:

- `metastable_plateau`
- `slow_dissolution`
- `slow_growth`
- `runaway`

based on long-time trends in:

- `eta_max(t)`
- `eta_integral(t)`
- `V_h(t)`
- `R_eff_h(t)`

The key scientific conclusion will be whether mature `observed_gp_diffuse` GP zones converge to a finite stable state or eventually dissolve.
