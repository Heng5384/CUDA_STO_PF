# Dry-Run Cleanup Plan

Date: 2026-05-17
Mode: dry-run only
Status: no move/delete/archive executed

## Scope

This plan covers local cleanup under the current Git workspace and a provisional cluster cleanup plan.

Important constraints:
- Keep the original constrained workflow data source intact.
- Do not delete any old result directory until a formal consolidated output directory is created and verified.
- Prefer moving old versions into `archive/cleanup_20260517/` rather than deleting.
- Prefer the existing `tools/analysis/` directory as the authoritative analysis-tool location instead of creating a second competing root-level `analysis/` tree.

## Findings Summary

### Local

Observed temporary result directories:
- `energy_component_audit_T400_xB0p050_v1`
- `energy_component_audit_T400_xB0p050_v2`
- `energy_component_audit_T400_xB0p050_v3`
- `nucleation_rate_outputs_T400_xB0p050`
- `nucleation_rate_outputs_T400_xB0p050_audit_diag_v2`
- `nucleation_rate_outputs_T400_xB0p050_refB_v1`
- `nucleation_rate_outputs_T400_xB0p050_refB_v2`
- `nucleation_rate_outputs_T400_xB0p050_refaudit_v1`
- `nucleation_rate_outputs_T400_xB0p050_v2`
- `nucleation_rate_outputs_T400_xB0p050_v3`
- `nucleation_rate_outputs_T400_xB0p050_v4`
- `nucleation_rate_outputs_T400_xB0p050_v5`
- `nucleation_rate_outputs_T400_xB0p050_v6`
- `Results_workflow_T400_xB0p050_cnt_scan_sync`

Observed analysis scripts:
- root: `compute_explicit_nucleation_rates.py`
- root: `energy_component_barrier_audit.py`
- already in `tools/analysis/`: `summarize_cnt_scan_from_guide.py`
- already in `tools/analysis/`: `AR_analysis.py`

Observed completeness:
- `energy_component_audit_T400_xB0p050_v1/v2/v3` are each structurally complete.
- `energy_component_audit_T400_xB0p050_v3` is the newest and should be the retained authoritative energy-component audit.
- None of the current `nucleation_rate_outputs_T400_xB0p050*` directories alone contain both:
  - `current_results_master_table_fitted.csv`
  - `reference_energy.csv`
- Those two authoritative same-strain-reference inputs currently live in `Results_workflow_T400_xB0p050_cnt_scan_sync/`.
- `nucleation_rate_outputs_T400_xB0p050_refB_v2` is the best candidate for the retained nucleation-rate output directory because it contains:
  - `nucleation_rate_table_diagnostic.csv`
  - `reference_energy_audit.csv`
  - `barrier_reference_comparison.csv`
  - `nucleation_rate_metadata.json`
  - plots
- Therefore the local nucleation-rate result is currently split across two locations and should be consolidated before any archival cleanup.

### Cluster

Observed temporary result directories:
- `energy_component_audit_T400_xB0p050_v1`
- `energy_component_audit_T400_xB0p050_v2`
- `energy_component_audit_T400_xB0p050_v3`
- `nucleation_rate_outputs_T400_xB0p050`
- `nucleation_rate_outputs_T400_xB0p050_audit_diag`
- `nucleation_rate_outputs_T400_xB0p050_audit_diag_v2`
- `nucleation_rate_outputs_T400_xB0p050_refB_v1`
- `nucleation_rate_outputs_T400_xB0p050_refB_v2`
- `nucleation_rate_outputs_T400_xB0p050_refaudit_v1`
- `nucleation_rate_outputs_T400_xB0p050_v2`
- `nucleation_rate_outputs_T400_xB0p050_v3`
- `nucleation_rate_outputs_T400_xB0p050_v4`
- `nucleation_rate_outputs_T400_xB0p050_v5`
- `nucleation_rate_outputs_T400_xB0p050_v6`
- `Results/workflows/T400_xB0p050`

Observed analysis scripts:
- root: `compute_explicit_nucleation_rates.py`
- root: `energy_component_barrier_audit.py`
- root: `summarize_cnt_scan_from_guide.py`
- root: `tools/analysis/summarize_cnt_scan_from_guide.py`

Observed completeness:
- `Results/workflows/T400_xB0p050/cnt_scan/current_results_master_table_fitted.csv` exists.
- `Results/workflows/T400_xB0p050/cnt_scan/reference_energy.csv` exists.
- `energy_component_audit_T400_xB0p050_v1/v2/v3` are each structurally complete.
- `energy_component_audit_T400_xB0p050_v3` is the newest and should be the retained authoritative energy-component audit.
- None of the current `nucleation_rate_outputs_T400_xB0p050*` directories alone contain both:
  - `current_results_master_table_fitted.csv`
  - `reference_energy.csv`
- On cluster those two authoritative same-strain-reference inputs already live in the original workflow tree:
  - `Results/workflows/T400_xB0p050/cnt_scan/current_results_master_table_fitted.csv`
  - `Results/workflows/T400_xB0p050/cnt_scan/reference_energy.csv`
- `nucleation_rate_outputs_T400_xB0p050_refB_v2` is the best candidate for the retained nucleation-rate output directory because it contains:
  - `nucleation_rate_table_diagnostic.csv`
  - `reference_energy_audit.csv`
  - `barrier_reference_comparison.csv`
  - `nucleation_rate_metadata.json`
  - plots
- The cluster root contains a duplicate `summarize_cnt_scan_from_guide.py` in addition to `tools/analysis/summarize_cnt_scan_from_guide.py`; that duplicate should be marked `needs_review` before any move/archive.

## A. Suggested Formal Directories To Keep

### Local keep

1. `Results/workflows/T400_xB0p050` if present locally in the future, or the remote workflow root on cluster.
Reason:
- original constrained workflow source data
- must not be deleted

2. `Results_workflow_T400_xB0p050_cnt_scan_sync`
Reason:
- currently holds the authoritative `current_results_master_table_fitted.csv` and `reference_energy.csv`
- these are required to define the formal same-strain-reference barrier
- should be kept until consolidated into a formal output directory

3. `nucleation_rate_outputs_T400_xB0p050_refB_v2`
Reason:
- newest same-strain-reference-aware nucleation-rate output set among the local `refB` directories
- contains the latest diagnostic tables and barrier-reference comparison outputs
- should remain as the source for the formal retained nucleation-rate output

4. `energy_component_audit_T400_xB0p050_v3`
Reason:
- newest and corrected energy-component barrier audit
- includes repaired fallback-reference handling and complete component `DeltaG_*` output

### Recommended formal post-cleanup target layout

Use:
- `tools/analysis/`
- `outputs/T400_xB0p050/nucleation_rate/`
- `outputs/T400_xB0p050/energy_component_barrier/`
- `archive/cleanup_20260517/`

Reason:
- `tools/analysis/` already exists and matches repo conventions
- introducing a new root `analysis/` directory would duplicate the existing analysis namespace
- `outputs/` cleanly separates curated result bundles from raw workflow directories and one-off temporary output folders

## B. Suggested Old-Version Directories To Archive

Archive target:
- `archive/cleanup_20260517/`

### Local archive candidates after formal consolidation is verified

Energy-component audits:
- `energy_component_audit_T400_xB0p050_v1`
  - older complete version superseded by `v3`
- `energy_component_audit_T400_xB0p050_v2`
  - older complete version superseded by `v3`

Nucleation-rate outputs:
- `nucleation_rate_outputs_T400_xB0p050`
  - early incomplete base output
- `nucleation_rate_outputs_T400_xB0p050_audit_diag_v2`
  - older diagnostic stage before same-strain reference workflow was finalized
- `nucleation_rate_outputs_T400_xB0p050_refaudit_v1`
  - intermediate reference audit stage superseded by `refB_v2`
- `nucleation_rate_outputs_T400_xB0p050_refB_v1`
  - older same-strain-reference output superseded by `refB_v2`
- `nucleation_rate_outputs_T400_xB0p050_v2`
  - older incomplete stage
- `nucleation_rate_outputs_T400_xB0p050_v3`
  - older incomplete stage
- `nucleation_rate_outputs_T400_xB0p050_v4`
  - older incomplete stage
- `nucleation_rate_outputs_T400_xB0p050_v5`
  - older incomplete stage
- `nucleation_rate_outputs_T400_xB0p050_v6`
  - older pre-reference-consolidation stage

### Conditional archive candidate

- `nucleation_rate_outputs_T400_xB0p050_refB_v2`
Reason:
- archive only after its retained files have been copied or moved into the formal curated directory `outputs/T400_xB0p050/nucleation_rate/` and verified.

- `Results_workflow_T400_xB0p050_cnt_scan_sync`
Reason:
- archive only after `current_results_master_table_fitted.csv` and `reference_energy.csv` have been consolidated into the formal curated directory and verified.
- alternatively keep it permanently if you want a lightweight synced CNT-summary bundle independent of `outputs/`.

## C. Suggested Directories Safe To Delete

Current dry-run recommendation:
- no result directory should be deleted directly at this stage

Reason:
- older result directories are not empty
- several contain intermediate audit artifacts worth preserving in `archive/`
- the current authoritative nucleation-rate result is split across two local directories, so deleting any source now is risky

Possible future safe-delete-only candidates after manual review:
- empty cache folders such as temporary `.tmp_*` or `__pycache__` style directories, if confirmed unrelated to active workflows
- not included in this cleanup batch by default

## D. Suggested Python Script Moves Into Authoritative Analysis Directory

Preferred authoritative analysis directory:
- `tools/analysis/`

### Move into `tools/analysis/`

1. `compute_explicit_nucleation_rates.py`
   -> `tools/analysis/compute_explicit_nucleation_rates.py`
Reason:
- root-level one-off analysis tool should live with other workflow analysis scripts
- avoids scattering analysis entry points across the repo root

2. `energy_component_barrier_audit.py`
   -> `tools/analysis/energy_component_barrier_audit.py`
Reason:
- same rationale as above
- tightly coupled to the constrained-workflow summary scripts already in `tools/analysis/`

### Keep in place inside `tools/analysis/`

3. `tools/analysis/summarize_cnt_scan_from_guide.py`
Reason:
- already correctly located

4. `tools/analysis/AR_analysis.py`
Reason:
- already correctly located

### Not found locally in this dry-run

5. `barrier_volume_normalization_audit.py`
Reason:
- not present; no move planned

6. `reference_energy_audit.py`
Reason:
- not present; no move planned

### Additional documentation file to create during the actual cleanup pass

- `tools/analysis/README_analysis_tools.md`
Reason:
- repo already uses `tools/analysis/`
- documentation should live next to the scripts it describes

## E. Cluster Directories Requiring Equivalent Cleanup

Cluster candidates inspected and later to align with local cleanup:
- `Results/workflows/T400_xB0p050`
- `nucleation_rate_outputs_T400_xB0p050*`
- `energy_component_audit_T400_xB0p050*`
- root-level `compute_explicit_nucleation_rates.py`
- root-level `energy_component_barrier_audit.py`
- root-level `summarize_cnt_scan_from_guide.py`
- `tools/analysis/summarize_cnt_scan_from_guide.py`
- `tools/analysis/AR_analysis.py`

Expected cluster formal target layout:
- `tools/analysis/`
- `outputs/T400_xB0p050/nucleation_rate/`
- `outputs/T400_xB0p050/energy_component_barrier/`
- `archive/cleanup_20260517/`

Cluster restrictions for the actual cleanup pass:
- do not remove `Results/workflows/T400_xB0p050`
- do not remove raw `energy_minimize_*.csv`
- do not remove generated params, source params, or raw summaries
- do not remove slurm logs unless individually confirmed duplicate and empty
- prefer `mv` to `archive/cleanup_20260517/` rather than deletion
- mark the root-level cluster `summarize_cnt_scan_from_guide.py` as `needs_review` before any move because there are two copies on the remote repo

## F. Decision Rationale By Item

### Why `energy_component_audit_T400_xB0p050_v3` is the keeper
- newest complete version
- includes repaired fallback-reference supplementation
- matches the final physical interpretation that surface contribution dominates the excess barrier

### Why `nucleation_rate_outputs_T400_xB0p050_refB_v2` is the keeper candidate
- latest same-strain-reference-aware nucleation-rate diagnostic result
- contains the reference-comparison outputs missing from earlier directories
- however it is not self-contained because the authoritative `current_results_master_table_fitted.csv` and `reference_energy.csv` still live elsewhere
  - locally: `Results_workflow_T400_xB0p050_cnt_scan_sync/`
  - on cluster: `Results/workflows/T400_xB0p050/cnt_scan/`

### Why older `nucleation_rate_outputs_*` should be archived rather than deleted
- they document intermediate stages: pre-reference, audit-only, ref-audit, early diagnostic
- they are useful for provenance and rollback
- they are superseded for day-to-day use

### Why `Results_workflow_T400_xB0p050_cnt_scan_sync` must be preserved for now
- contains the same-strain-reference CNT summary inputs required to interpret the formal barrier
- current latest nucleation-rate directory is incomplete without it

### Why `tools/analysis/` should be used instead of a new root `analysis/`
- existing repo convention already groups workflow analysis scripts there
- moving tools into `tools/analysis/` minimizes path churn and import breakage
- a second top-level `analysis/` would create two competing homes for analysis tooling

### Why the cluster root-level `summarize_cnt_scan_from_guide.py` needs review
- there is already an authoritative copy under `tools/analysis/`
- the extra root-level copy may be stale or manually copied
- do not archive or delete it until contents are compared

## Recommended Execution Order For The Actual Cleanup Pass

1. Create `outputs/T400_xB0p050/nucleation_rate/`.
2. Populate it from:
   - `Results_workflow_T400_xB0p050_cnt_scan_sync/current_results_master_table_fitted.csv`
   - `Results_workflow_T400_xB0p050_cnt_scan_sync/reference_energy.csv`
   - `nucleation_rate_outputs_T400_xB0p050_refB_v2/*`
3. Verify all required nucleation-rate files exist in the new formal directory.
4. Create `outputs/T400_xB0p050/energy_component_barrier/`.
5. Populate it from `energy_component_audit_T400_xB0p050_v3/`.
6. Move `compute_explicit_nucleation_rates.py` and `energy_component_barrier_audit.py` into `tools/analysis/`.
7. Add `tools/analysis/README_analysis_tools.md`.
8. Run `--help` smoke tests from repo root.
9. Move superseded result directories into `archive/cleanup_20260517/`.
10. Only after all local checks pass, repeat the same structure on cluster.

## Local And Cluster Keep/Archive Summary

### LOCAL keep now
- `Results_workflow_T400_xB0p050_cnt_scan_sync`
- `nucleation_rate_outputs_T400_xB0p050_refB_v2`
- `energy_component_audit_T400_xB0p050_v3`
- `tools/analysis/summarize_cnt_scan_from_guide.py`
- `tools/analysis/AR_analysis.py`
- root `compute_explicit_nucleation_rates.py` until moved
- root `energy_component_barrier_audit.py` until moved

### CLUSTER keep now
- `Results/workflows/T400_xB0p050`
- `nucleation_rate_outputs_T400_xB0p050_refB_v2`
- `energy_component_audit_T400_xB0p050_v3`
- `tools/analysis/summarize_cnt_scan_from_guide.py`
- root `compute_explicit_nucleation_rates.py` until moved
- root `energy_component_barrier_audit.py` until moved

### CLUSTER needs review
- root `summarize_cnt_scan_from_guide.py`

## Dry-Run Conclusion

The cleanup can proceed safely only after a formal curated `outputs/T400_xB0p050/` tree is created and verified. Old versions should then be archived, not deleted. Both local and cluster inventories show the same structural issue: the authoritative nucleation-rate result is split between the nucleation output directory and the CNT summary/reference CSV source.
