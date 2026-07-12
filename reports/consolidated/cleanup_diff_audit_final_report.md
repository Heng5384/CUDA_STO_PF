# Cleanup Diff Audit Final Report

## 1. Executive Summary

- This worktree is still dirty, but the change surface is now cleanly split into cleanup archive, path migration, functional nucleation code, and generated outputs.
- The dominant functional delta remains `main_cuda.cu` / `pf_params.h`; cleanup artifacts live under `reports/consolidated/` and `reports/root_archive/`.
- Root report outputs are gone; only root entry docs remain.

## 2. Root Cleanliness Status

- Root `.md` / `.csv` / `.txt` outputs remaining: none.
- Remaining root docs: `README.md`, `MIGRATION_NOTES.md`.
- No new script in this audit is writing report outputs back to root as a primary destination.

## 3. Worktree Diff Categories

- cleanup_archive: 41
- default_path_migration: 7
- functional_nucleation_changes: 28
- generated_runtime_outputs: 108
- uncertain: 0

## 4. Cleanup-Related Files

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
- `reports/root_archive/csv/thermo_object_inventory.csv`
- `reports/root_archive/md/beta_phase_report.md`
- `reports/root_archive/md/composition_mapping_risk_report.md`
- `reports/root_archive/md/coupling_consistency_report.md`
- `reports/root_archive/md/cuda_runtime_fix_report.md`
- `reports/root_archive/md/discretization_bias_report.md`
- `reports/root_archive/md/dual_track_architecture.md`
- `reports/root_archive/md/gp_physics_report.md`
- `reports/root_archive/md/mapping_binding_final_report.md`
- `reports/root_archive/md/microstructure_capability_report.md`

## 5. Path-Migration-Related Files

- `README.md`
- `analyze_nucleation_event_observables.py`
- `normalize_nucleation_physical_units.py`
- `nucleus_selector.py`
- `reports/final_physics_validation_report.md`
- `validate_minimal_physics_loop.py`
- `validate_nucleation_theory_vs_cuda.py`

## 6. Functional Changes Separated from Cleanup

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
- `scripts/dry_run_gp_mass_budget_initialization.py`
- `scripts/gp_assisted_locked_S_temperature_prediction_template.py`
- `scripts/propose_gp_S_calibration_and_prediction.py`
- `shape_library/README.md`
- `tools/analysis/gp_assisted_beta_nucleation.py`
- `tools/analysis/run_gp_external_prediction_validation.py`
- `tools/analysis/validate_cuda_nucleus_usability.py`
- `tools/analysis/workstation_cnt_batch.py`

## 7. Generated Outputs and Tracking Guidance

- These are mostly reports, tables, validation outputs, and generated config artifacts.
- They are safe to track if the intent is to preserve calibration history, but they should stay separate from the code commit.

### Representative generated outputs

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
- `reports/data/step41d_gp_beta_driving_forces/step41D_beta_drive_vs_xB_by_T.png`
- `reports/data/step41d_gp_beta_driving_forces/step41D_drive_curves.csv`
- `reports/data/step41d_gp_beta_driving_forces/step41D_drive_sample_points.csv`
- `reports/data/step41d_gp_beta_driving_forces/step41D_threshold_xB_vs_T.png`
- `reports/gp/external_prediction_validation.csv`
- `reports/gp/gp_validation_report.csv`
- `reports/gp_assisted_beta_mass_ledger/cuda_patch_proposal_gp_assisted_mass_conserving_beta_nucleation.md`
- `reports/gp_assisted_beta_mass_ledger/existing_beta_insertion_functions.csv`
- `reports/gp_assisted_beta_mass_ledger/existing_beta_insertion_path.md`
- `reports/gp_assisted_beta_mass_ledger/gp_assisted_beta_mass_ledger_next_step_report.md`

## 8. Risk Assessment

- High: `main_cuda.cu`, `pf_params.h`.
- Medium: `jobs/_site_env.sh`, `nucleus_*` workflow code, `scripts/*`, `tools/analysis/*`.
- Low: `reports/root_archive/**`, `reports/consolidated/root_*`, generated reports/data/configs.

## 9. Recommended Commit Grouping

- Commit 1: cleanup archive.
- Commit 2: path migration.
- Commit 3: functional nucleation workflow.
- Commit 4: generated results and reports.

## 10. Next Action

- Review the commit plan, then stage cleanup and functional work separately so the cleanup diff stays readable.
- If you want, the next operational step is to stage Commit 1 first and keep the remaining groups untouched.
