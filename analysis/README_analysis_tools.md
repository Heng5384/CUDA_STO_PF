# Analysis Tools

This directory contains the curated entry-point scripts for the constrained-nucleus,
same-strain-reference, and explicit nucleation-rate workflow.

The original constrained workflow data directory must not be deleted:

- `Results/workflows/T400_xB0p050` on cluster

The formal barrier must use the same-strain matrix-only reference-subtracted
quantity:

- `F_CNT_peak_hat_excess`

Do not use absolute `F_CNT_peak_hat` as the formal nucleation barrier.

`Z_r` fallback is diagnostic only. It must not be treated as the strict physical
nucleation rate.

## Scripts

### `analysis/compute_explicit_nucleation_rates.py`

Purpose:
- compute explicit nucleation-rate tables from constrained CNT summaries,
  same-strain references, and PF/CUDA-consistent parameters
- generate diagnostic and strict rate tables, barrier audits, and plots

Primary inputs:
- constrained workflow root such as `Results/workflows/T400_xB0p050`
- `current_results_master_table_fitted.csv`
- `reference_energy.csv`
- optional profile and log files

Primary outputs:
- `nucleation_rate_table.csv`
- `nucleation_rate_table_diagnostic.csv`
- `reference_energy_audit.csv`
- `barrier_reference_comparison.csv`
- `nucleation_rate_metadata.json`
- plots

Default output behavior:
- when `--root` points directly to a workflow directory such as
  `Results/workflows/T400_xB0p050`, output is written to
  `Results/workflows/T400_xB0p050/analysis/nucleation_rate/`

Example:

```bash
python analysis/compute_explicit_nucleation_rates.py \
  --root Results/workflows/T400_xB0p050 \
  --pattern "cntcon_*" \
  --diffusivity-mode arrhenius_ag \
  --theta steady
```

### `analysis/energy_component_barrier_audit.py`

Purpose:
- build same-strain-reference-subtracted component profiles
- separate surface, chemical, and elastic barrier contributions at the saddle
  and along the full constrained radius scan

Primary inputs:
- workflow root such as `Results/workflows/T400_xB0p050`
- `guide_cnt_scan.csv`
- `current_results_master_table_fitted.csv`
- `reference_energy.csv`
- `energy_minimize_*.csv`
- `summary.txt`

Primary outputs:
- `energy_component_barrier_audit.csv`
- `energy_profile_excess_by_case.csv`
- `energy_component_barrier_metadata.json`
- plots

Default output behavior:
- when `--out` is omitted, output is written to
  `Results/workflows/<case>/analysis/energy_component_barrier/`

Example:

```bash
python analysis/energy_component_barrier_audit.py \
  --workflow-root Results/workflows/T400_xB0p050 \
  --repo-root .
```

## Related Workflow Helpers

These remain in the existing repo analysis namespace because other workflow code
already imports them there:

- `tools/analysis/summarize_cnt_scan_from_guide.py`
- `tools/analysis/AR_analysis.py`

## Recommended Run Order

1. Run or collect the constrained radius scan.
2. Generate the same-strain matrix-only `reference_energy.csv`.
3. Summarize the CNT scan with same-strain reference subtraction.
4. Compute the explicit nucleation diagnostic rate.
5. Audit the energy components.
6. Generate and review plots.

## Notes

- The authoritative barrier is the same-strain-reference-subtracted
  `F_CNT_peak_hat_excess`.
- Do not use absolute `F_CNT_peak_hat` as the final barrier.
- `Z_r` fallback is allowed only for diagnostic rate output.
- The original constrained workflow directory is source data and must not be
  deleted.
- The curated analysis outputs now live inside the workflow under
  `Results/workflows/<case>/analysis/`.
