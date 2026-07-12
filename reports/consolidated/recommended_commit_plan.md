# Recommended Commit Plan

## Commit 1

`repo cleanup: archive root reports and generated tables under reports/`

### Suggested files

- `reports/root_archive/**`
- `reports/consolidated/root_*`
- `reports/consolidated/pre_cleanup_root_inventory.txt`
- `reports/maintenance/cleanup_acceptance_report.md`
- `reports/maintenance/cleanup_operations.log`
- `reports/maintenance/dry_run_cleanup_plan.md`

## Commit 2

`path migration: route validation and nucleation outputs to reports/`

### Suggested files

- `README.md`
- `validate_minimal_physics_loop.py`
- `normalize_nucleation_physical_units.py`
- `validate_nucleation_theory_vs_cuda.py`
- `analyze_nucleation_event_observables.py`
- `nucleus_selector.py`
- `reports/final_physics_validation_report.md`

## Commit 3

`nucleation workflow: dynamic-continue / nucleus generator / CNT changes`

### Suggested files

- `main_cuda.cu`
- `pf_params.h`
- `jobs/_site_env.sh`
- `cnt_scan_pipeline.py`
- `nucleus_geometry_mapper.py`
- `nucleus_orchestrator.py`
- `pf_dynamics_core.py`
- `s_field_definition.py`
- `plot_step41D_gp_beta_driving_forces.py`
- `shape_library/README.md`
- `nucleus_generator/**`
- `nucleus_parametric_generator/**`
- `scripts/**`
- `tools/analysis/**`

## Commit 4

`results: add selected validation outputs and reports`

### Suggested files

- `reports/validation/**`
- `reports/nucleation/**`
- `reports/cnt/**`
- `reports/gp/**`
- `reports/stochastic_layer/**`
- `reports/data/**`
- `reports/nucleus_cuda_integration/**`
- `reports/gp_assisted_*/**`
- `reports/step_reports/STEP41D_GP_BETA_DRIVING_FORCE_COMPARISON_REPORT.md`
- `CNT_SCAN_WORKSTATION_RUN/**`
- `coupling_interface.json`
- `nucleus_catalog.json`
- `nucleus_catalog.schema.json`
- `physical_units_config.json`
- `selected_nucleus.json`
- `reports/consolidated/current_worktree_audit.md`
- `reports/consolidated/current_worktree_audit.csv`
- `reports/consolidated/root_cleanliness_final_check.md`
- `reports/consolidated/default_path_migration_final_check.md`
- `reports/consolidated/recommended_commit_plan.md`
- `reports/consolidated/cleanup_diff_audit_final_report.md`

## Manual review

- none

## Notes

- Keep cleanup archive and path migration out of the functional nucleation commit.
- Generated reports/data are best added after the code paths stabilize, so reviewers can compare code and outputs separately.
