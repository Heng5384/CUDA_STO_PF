# Cleanup Acceptance Report

Date: 2026-05-17

## A. Final Retained Directory Tree

### LOCAL

```text
analysis/
  README_analysis_tools.md
  compute_explicit_nucleation_rates.py
  energy_component_barrier_audit.py

Results/
  workflows/
    T400_xB0p050/
      analysis/
        nucleation_rate/
          barrier_conversion_audit.csv
          barrier_reference_comparison.csv
          current_results_master_table_fitted.csv
          nucleation_rate_metadata.json
          nucleation_rate_table.csv
          nucleation_rate_table_diagnostic.csv
          reference_energy.csv
          reference_energy_audit.csv
          plots/
        energy_component_barrier/
          energy_component_barrier_audit.csv
          energy_component_barrier_metadata.json
          energy_profile_excess_by_case.csv
          plots/

archive/
  cleanup_20260517/
    ...
```

### CLUSTER

```text
/data/home/luozhiheng/CUDA_STO_PF/
  analysis/
    README_analysis_tools.md
    compute_explicit_nucleation_rates.py
    energy_component_barrier_audit.py
  archive/
    cleanup_20260517/
      ...
  Results/
    workflows/
      T400_xB0p050/
        analysis/
          nucleation_rate/
          energy_component_barrier/
        ...
```

## B. Archived Directories

### LOCAL archived into `archive/cleanup_20260517/`

- `Results_workflow_T400_xB0p050_cnt_scan_sync`
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

### CLUSTER archived into `archive/cleanup_20260517/`

- `compute_explicit_nucleation_rates.py` from repo root
- `energy_component_barrier_audit.py` from repo root
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

## C. Moved Scripts

### LOCAL

- `compute_explicit_nucleation_rates.py` -> `analysis/compute_explicit_nucleation_rates.py`
- `energy_component_barrier_audit.py` -> `analysis/energy_component_barrier_audit.py`

### CLUSTER

- repo-root `compute_explicit_nucleation_rates.py` -> authoritative copy now in `analysis/compute_explicit_nucleation_rates.py`
- repo-root `energy_component_barrier_audit.py` -> authoritative copy now in `analysis/energy_component_barrier_audit.py`

### Left In Existing Workflow Namespace

- `tools/analysis/summarize_cnt_scan_from_guide.py`
- `tools/analysis/AR_analysis.py`

Reason:
- they were already in the established workflow-analysis namespace
- other repo code imports them from there

## D. Required File Checks

### Analysis tools

- `analysis/compute_explicit_nucleation_rates.py` present
- `analysis/energy_component_barrier_audit.py` present
- `analysis/README_analysis_tools.md` present

### Formal nucleation-rate output

- `Results/workflows/T400_xB0p050/analysis/nucleation_rate/current_results_master_table_fitted.csv` present
- `Results/workflows/T400_xB0p050/analysis/nucleation_rate/reference_energy.csv` present
- `Results/workflows/T400_xB0p050/analysis/nucleation_rate/nucleation_rate_table_diagnostic.csv` present
- `Results/workflows/T400_xB0p050/analysis/nucleation_rate/reference_energy_audit.csv` present
- `Results/workflows/T400_xB0p050/analysis/nucleation_rate/barrier_reference_comparison.csv` present
- `Results/workflows/T400_xB0p050/analysis/nucleation_rate/nucleation_rate_metadata.json` present

### Formal energy-component output

- `Results/workflows/T400_xB0p050/analysis/energy_component_barrier/energy_component_barrier_audit.csv` present
- `Results/workflows/T400_xB0p050/analysis/energy_component_barrier/energy_profile_excess_by_case.csv` present
- `Results/workflows/T400_xB0p050/analysis/energy_component_barrier/energy_component_barrier_metadata.json` present

## E. Smoke Tests

### LOCAL

Passed:

```bash
python3 analysis/compute_explicit_nucleation_rates.py --help
python3 analysis/energy_component_barrier_audit.py --help
```

### CLUSTER

Passed:

```bash
python3 analysis/compute_explicit_nucleation_rates.py --help
python3 analysis/energy_component_barrier_audit.py --help
```

## F. Cluster Sync Status

Cluster cleanup was executed conservatively and is now aligned with the local
curated structure:

- `analysis/` created and populated
- `Results/workflows/T400_xB0p050/analysis/nucleation_rate/` created and populated
- `Results/workflows/T400_xB0p050/analysis/energy_component_barrier/` created and populated
- superseded root outputs moved into `archive/cleanup_20260517/`
- original workflow source `Results/workflows/T400_xB0p050` preserved intact

Remaining remote review item:

- root-level `summarize_cnt_scan_from_guide.py`

Status:
- not moved
- not deleted
- still marked `needs_review` because `tools/analysis/summarize_cnt_scan_from_guide.py`
  also exists on cluster

## Notes

- The formal nucleation barrier must use same-strain-reference-subtracted
  `F_CNT_peak_hat_excess`.
- Absolute `F_CNT_peak_hat` remains diagnostic only.
- `Z_r` fallback remains diagnostic only.
- The original constrained workflow directory must not be deleted.
