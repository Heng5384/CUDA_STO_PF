# Current Worktree Audit

## Executive Summary
- Top-level `git status --short` entries: 57
- File-level dirty paths audited: 184
- Tracked modified files: 4
- Untracked files/directories: 180

## Category Counts
- cleanup_archive: 41
- default_path_migration: 7
- functional_nucleation_changes: 28
- generated_runtime_outputs: 108
- uncertain: 0

## Recommended Commit Groups
- Commit 1 — repo cleanup: archive root reports and generated tables under reports/: 41 files
- Commit 2 — path migration: route validation and nucleation outputs to reports/: 7 files
- Commit 3 — nucleation workflow: dynamic-continue / nucleus generator / CNT changes: 28 files
- Commit 4 — results: add selected validation outputs and reports: 108 files
- manual_review: 0 files

## Cleanup Archive Highlights
- `reports/consolidated/pre_cleanup_root_inventory.txt`
- `reports/consolidated/root_cleanliness_final_check.md`
- `reports/consolidated/root_cleanup_master_index.csv`
- `reports/consolidated/root_cleanup_report.md`
- `reports/consolidated/root_csv_archive_index.csv`
- `reports/consolidated/root_csv_archive_summary.md`
- `reports/consolidated/root_default_path_migration_index.csv`
- `reports/consolidated/root_default_path_migration_report.md`
- `reports/consolidated/root_folder_archive_index.csv`
- `reports/consolidated/root_level_reports_consolidated.md`
- `reports/consolidated/root_retained_path_migration_plan.csv`
- `reports/consolidated/root_retained_path_migration_plan.md`
- `reports/consolidated/root_retained_reference_index.csv`
- `reports/consolidated/root_retained_reference_index.md`
- `reports/root_archive/csv/composition_variable_audit.csv`
- `reports/root_archive/csv/cuda_binding_verification.csv`
- `reports/root_archive/csv/cuda_execution_summary.csv`
- `reports/root_archive/csv/mapping_ambiguity_report.csv`
- `reports/root_archive/csv/physics_consistency_delta.csv`
- `reports/root_archive/csv/reference_state_audit.csv`

## Path Migration Highlights
- `README.md`
- `analyze_nucleation_event_observables.py`
- `normalize_nucleation_physical_units.py`
- `nucleus_selector.py`
- `reports/final_physics_validation_report.md`
- `validate_minimal_physics_loop.py`
- `validate_nucleation_theory_vs_cuda.py`

## Functional Changes Highlights
- `jobs/_site_env.sh`
- `main_cuda.cu`
- `pf_params.h`
- `cnt_scan_pipeline.py`
- `nucleus_generator/cuda_nucleus_builder.h`
- `nucleus_generator/nucleus_generation_report.md`
- `nucleus_generator/nucleus_generator.cpp`
- `nucleus_generator/nucleus_resampler.py`
- `nucleus_geometry_mapper.py`
- `nucleus_orchestrator.py`
- `nucleus_parametric_generator/acceptance_summary.json`
- `nucleus_parametric_generator/cuda_nucleus_eval.cu`
- `nucleus_parametric_generator/dynamic_continue_parser.cpp`
- `nucleus_parametric_generator/parametric_nucleus_builder.cu`
- `nucleus_parametric_generator/validation_report.md`
- `nucleus_parametric_generator/workstation_test_runner.sh`
- `pf_dynamics_core.py`
- `plot_step41D_gp_beta_driving_forces.py`
- `s_field_definition.py`
- `scripts/dry_run_gp_assisted_beta_event.py`

## Generated Outputs Highlights
- `CNT_SCAN_WORKSTATION_RUN/cuda_nucleus_compatibility/cuda_insertion_compatibility_table.csv`
- `CNT_SCAN_WORKSTATION_RUN/cuda_nucleus_compatibility/cuda_nucleus_usability_report.md`
- `CNT_SCAN_WORKSTATION_RUN/cuda_nucleus_compatibility/geometry_validity_report.csv`
- `CNT_SCAN_WORKSTATION_RUN/cuda_nucleus_compatibility/physics_consistency_report.csv`
- `CNT_SCAN_WORKSTATION_RUN/cuda_nucleus_templates/metadata.json`
- `coupling_interface.json`
- `nucleus_catalog.json`
- `nucleus_catalog.schema.json`
- `physical_units_config.json`
- `reports/cnt/cnt_prediction_table.csv`
- `reports/cnt/normalized_cnt_prediction.csv`
- `reports/cnt/nucleation_barrier_observed.csv`
- `reports/cnt/nucleation_observation_table.csv`
- `reports/cnt/nucleation_prediction_table.csv`
- `reports/consolidated/cleanup_diff_audit_final_report.md`
- `reports/consolidated/current_worktree_audit.csv`
- `reports/consolidated/current_worktree_audit.md`
- `reports/consolidated/default_path_migration_final_check.md`
- `reports/consolidated/recommended_commit_plan.md`
- `reports/data/cuda_nucleation_observation.csv`
- `reports/data/cuda_nucleation_spatial_density.csv`
- `reports/data/cuda_nucleation_time_histogram.csv`
- `reports/data/normalized_nucleation_rate.csv`
- `reports/data/step41d_gp_beta_driving_forces/step41D_GP_drive_vs_xB_by_T.png`
- `reports/data/step41d_gp_beta_driving_forces/step41D_GP_vs_beta_drive_by_T.png`

## Uncertain Paths

## Notes
- Full per-path classification is in `current_worktree_audit.csv`.
- This audit includes the six `reports/consolidated/*` deliverables created during the task as generated outputs.
- This pass does not change files outside the generated reports below.
