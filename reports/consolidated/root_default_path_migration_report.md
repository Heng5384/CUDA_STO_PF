# Root Default Path Migration Report

## Executive Summary
All retained root markdown and CSV outputs were migrated into `reports/` subdirectories. The scripts that consume them now prefer the new locations and fall back to the legacy root paths when needed, so the move is reversible and low risk.

## Which Root Files Were Still Retained and Why
None of the target root markdown/CSV outputs remain in the repository root after this pass. The remaining root files are code, configuration, input JSON, or launch scripts and were outside the scope of this migration.

## Which Scripts Referenced Them
- `validate_minimal_physics_loop.py`
- `normalize_nucleation_physical_units.py`
- `validate_nucleation_theory_vs_cuda.py`
- `analyze_nucleation_event_observables.py`
- `nucleus_selector.py`
- `reports/final_physics_validation_report.md`
- `README.md`

## Which Default Paths Were Changed
- Validation outputs now default under `reports/validation/`
- Nucleation observables now default under `reports/nucleation/`
- CNT sweep tables now default under `reports/cnt/`
- GP validation outputs now default under `reports/gp/`
- Raw / normalized CSV data now default under `reports/data/`

## New Reports Directory Structure
- `reports/validation/`
- `reports/nucleation/`
- `reports/cnt/`
- `reports/gp/`
- `reports/data/`
- `reports/consolidated/`

## Backward Compatibility Behavior
Each updated script now prefers the new `reports/...` path and falls back to the old root path if the new file is absent. New writes go to `reports/` only; the root path is kept as a read fallback so older runs do not break during transition.

## Files Moved After Migration
- `comparison_summary.md` -> `reports/validation/comparison_summary.md`
- `nucleation_physics_observables_report.md` -> `reports/nucleation/nucleation_physics_observables_report.md`
- `nucleus_energy_output_index.md` -> `reports/nucleation/nucleus_energy_output_index.md`
- `nucleus_selector_integration_report.md` -> `reports/nucleation/nucleus_selector_integration_report.md`
- `quantitative_validation_summary.md` -> `reports/validation/quantitative_validation_summary.md`
- `validation_layer_report.md` -> `reports/validation/validation_layer_report.md`
- `validation_minimal_loop_report.md` -> `reports/validation/validation_minimal_loop_report.md`
- `cnt_prediction_table.csv` -> `reports/cnt/cnt_prediction_table.csv`
- `comparison_report.csv` -> `reports/validation/comparison_report.csv`
- `continuous_physics_summary.csv` -> `reports/validation/continuous_physics_summary.csv`
- `cuda_nucleation_observation.csv` -> `reports/data/cuda_nucleation_observation.csv`
- `cuda_nucleation_spatial_density.csv` -> `reports/data/cuda_nucleation_spatial_density.csv`
- `cuda_nucleation_time_histogram.csv` -> `reports/data/cuda_nucleation_time_histogram.csv`
- `external_prediction_validation.csv` -> `reports/gp/external_prediction_validation.csv`
- `gp_effect_analysis.csv` -> `reports/nucleation/gp_effect_analysis.csv`
- `gp_validation_report.csv` -> `reports/gp/gp_validation_report.csv`
- `normalized_cnt_prediction.csv` -> `reports/cnt/normalized_cnt_prediction.csv`
- `normalized_nucleation_rate.csv` -> `reports/data/normalized_nucleation_rate.csv`
- `nucleation_barrier_estimate.csv` -> `reports/nucleation/nucleation_barrier_estimate.csv`
- `nucleation_barrier_observed.csv` -> `reports/cnt/nucleation_barrier_observed.csv`
- `nucleation_event_log.csv` -> `reports/nucleation/nucleation_event_log.csv`
- `nucleation_observation_table.csv` -> `reports/cnt/nucleation_observation_table.csv`
- `nucleation_prediction_table.csv` -> `reports/cnt/nucleation_prediction_table.csv`
- `nucleation_rate_J.csv` -> `reports/nucleation/nucleation_rate_J.csv`
- `nucleus_shape_statistics.csv` -> `reports/nucleation/nucleus_shape_statistics.csv`
- `observed_barrier_reconstruction.csv` -> `reports/validation/observed_barrier_reconstruction.csv`
- `quantitative_validation_table.csv` -> `reports/validation/quantitative_validation_table.csv`
- `validation_comparison_table.csv` -> `reports/validation/validation_comparison_table.csv`

## Files Still Retained and Why
No target root markdown/CSV artifacts remain. Non-target files such as `README.md`, `MIGRATION_NOTES.md`, JSON inputs, code, and shell helpers remain in root because they are not report outputs.

## Validation Commands and Results
- `git status --short` run after migration; root target md/csv files no longer appear in the root file listing.
- `PASS (PYTHONPYCACHEPREFIX=/private/tmp/cpython-cache python3 -m py_compile validate_minimal_physics_loop.py normalize_nucleation_physical_units.py validate_nucleation_theory_vs_cuda.py analyze_nucleation_event_observables.py nucleus_selector.py)`
- `PASS (python3 --help for validate_minimal_physics_loop.py, normalize_nucleation_physical_units.py, nucleus_selector.py, validate_nucleation_theory_vs_cuda.py, analyze_nucleation_event_observables.py)`

## Risk Assessment
Low. The move only changed default output locations and preserved legacy input fallbacks. No physics model, CNT formula, or CUDA kernel code was modified.

## Rollback Instructions
1. Restore the default paths in the updated Python scripts back to the legacy root filenames.
2. Move the files from `reports/validation/`, `reports/nucleation/`, `reports/cnt/`, `reports/gp/`, and `reports/data/` back to the repository root.
3. Re-run the lightweight `py_compile` and `--help` checks.
4. If needed, delete the new `reports/...` copies only after the root versions are restored.

## Summary Counters
- retained_root_files_before: 28
- default_paths_migrated: 28
- files_moved_after_path_migration: 28
- files_still_retained: 0
- scripts_modified: 5
- py_compile_status: PASS (PYTHONPYCACHEPREFIX=/private/tmp/cpython-cache python3 -m py_compile validate_minimal_physics_loop.py normalize_nucleation_physical_units.py validate_nucleation_theory_vs_cuda.py analyze_nucleation_event_observables.py nucleus_selector.py)
- dry_run_status: PASS (python3 --help for validate_minimal_physics_loop.py, normalize_nucleation_physical_units.py, nucleus_selector.py, validate_nucleation_theory_vs_cuda.py, analyze_nucleation_event_observables.py)
- migration_report: reports/consolidated/root_default_path_migration_report.md
- migration_index: reports/consolidated/root_default_path_migration_index.csv
- git_status_after: dirty (expected; repository has pre-existing local modifications outside this migration)
- recommended_next_action: review the retained non-target root inputs/configs, then decide whether to normalize their paths in a follow-on pass
