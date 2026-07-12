# Nucleus Auto-Minimize Pipeline Audit

## Direct Answers

1. Does automatic nucleus-driven minimization exist?

   YES, but as a guide-driven batch workflow, not as a single autonomous controller. `rc_schur_nm` predictions generate `guide_cnt_scan.csv`; `jobs/submit_cnt_guide_serial.sbatch` automatically runs `main_cuda --mode=minimize` for each guide row.

2. Is scan stage connected to job submission?

   Partially YES. The connection is the `guide_cnt_scan.csv` contract plus the `GUIDE_CSV=... sbatch jobs/submit_cnt_guide_serial.sbatch` entry. The scan generator does not itself submit the job.

3. Or are they fully decoupled?

   Not fully decoupled. They share a concrete schema: `radius_nm`, `case_tag`, `physical_input_json_rel`, `raw_results_root_rel`, strain components, output paths. But they are operationally decoupled because a user or external workflow must invoke the Slurm submission and postprocessing stages.

4. What is the minimal missing link?

   A small orchestrator script, for example `run_cnt_auto_minimize_pipeline.py`, that calls:

   ```text
   setup_cnt_workflow.py
   -> sbatch submit_cnt_guide_serial.sbatch or local runner
   -> wait/check completion
   -> generate_cnt_geometry_summaries.py
   -> summarize_cnt_scan_from_guide.py
   -> compute_explicit_nucleation_rates.py
   -> optional nucleus_catalog rebuild / selector update
   ```

## Automatic Closed Loop Check

**prediction -> sweep -> minimize -> store -> reuse: NO as a fully automatic closed loop.**

What exists:

- prediction -> sweep: YES, in `tools/analysis/setup_cnt_workflow.py`.
- sweep -> minimize: YES after manual `sbatch`, in `jobs/submit_cnt_guide_serial.sbatch`.
- minimize -> store: YES, inside `main_cuda.cu`.
- store -> parsed summary: YES after manual postprocessing.
- parsed result -> reuse in later simulation: only partially. Recent `nucleus_selector.py` can read summary/rate outputs, but template availability and runtime insertion reuse are not fully closed for all predicted nuclei.

Missing links:

- No single script submits the generated guide automatically after prediction.
- No completion monitor automatically triggers geometry summaries and CNT/rate parsing.
- No guaranteed mapping from minimized critical nucleus output to CUDA insertion template for all entries.

## Key Pipeline Evidence

### Prediction / Candidate Generation

- `tools/analysis/compute_schur_rc_predictions.py`
  - `build_cases`, lines 93-103: enumerates strain modes and positive/negative strain cases.
  - `calc_elastic_energy_schur`, lines 166-197: computes constrained elastic energy.
  - `main`, lines 272-299: computes `DG_net` and `rc_schur_nm`.

- `tools/analysis/setup_cnt_workflow.py`
  - `_scan_radii`, lines 48-50: radius window around predicted `rc_nm`.
  - `main`, lines 116-127: computes Schur/CNT `rc_nm` per load case.
  - `main`, lines 221-281: emits scan candidates to `guide_cnt_scan.csv`.
  - `main`, lines 283-314: writes `schur_rc_predictions.csv`, `guide_cnt_scan.csv`, and workflow metadata.

### Minimize Job Submission

- `jobs/submit_cnt_guide_serial.sbatch`
  - lines 24-27: requires `GUIDE_CSV`.
  - lines 46-64: reads guide rows into per-row shell exports.
  - `run_case`, lines 184-258: builds and executes `./main_cuda --mode=minimize --minimize-full-model --radius-phys-nm "${radius_nm}"`.
  - lines 260-277: loops all guide rows.

- `jobs/submit_continue_minimize_guide_serial.sbatch`
  - `run_case`, lines 124-169: runs `./main_cuda --mode=minimize-continue` from selected continue sources.

### Result Storage / Parsing

- `main_cuda.cu`
  - lines 14219-14245: opens `energy_minimize_*.csv` and writes header.
  - lines 17602-17627: appends per-iteration energy/minimization rows including `F_total_CNT_hat`.

- `tools/analysis/generate_cnt_geometry_summaries.py`
  - `main`, lines 44-96: reads `guide_cnt_scan.csv` and `phi_final_*.vtk`, writes `summary.txt`.

- `tools/analysis/summarize_cnt_scan_from_guide.py`
  - lines 141-347: reads guide rows plus `energy_minimize_*.csv`, extracts peaks, writes `current_results_master_table_fitted.csv`.

- `analysis/compute_explicit_nucleation_rates.py`
  - `scan_case_dirs`, lines 854-920: finds case directories and local metadata.
  - lines 1840-1870 and 1881-1920: reads `energy_minimize_*.csv` and sibling energy points.
  - `write_outputs`, lines 3210-3364: writes nucleation-rate tables and metadata.

## Special Check: Predicted Radius Drives Minimize Jobs

YES, the predicted radius drives the generated minimize geometry set:

```text
rc_schur_nm
  -> _scan_radii(rc_schur_nm, window_nm, step_nm)
  -> guide_cnt_scan.csv: radius_nm
  -> submit_cnt_guide_serial.sbatch: --radius-phys-nm "${radius_nm}"
  -> main_cuda --mode=minimize
```

Precise locations:

- `tools/analysis/setup_cnt_workflow.py`, lines 116-127: prediction of `rc_nm`.
- `tools/analysis/setup_cnt_workflow.py`, lines 221-281: radius candidate rows.
- `jobs/submit_cnt_guide_serial.sbatch`, lines 221-242: per-row `main_cuda` minimize command.

## Related Scripts

### Nucleus scanning / prediction

- `tools/analysis/compute_schur_rc_predictions.py`
- `tools/analysis/setup_cnt_workflow.py`
- `tools/analysis/workflow_utils.py`
- `tools/analysis/analyze_cnt_peak_table.py`
- `tools/analysis/summarize_cnt_scan_from_guide.py`
- `analysis/compute_explicit_nucleation_rates.py`
- `analysis/energy_component_barrier_audit.py`
- `nucleus_selector.py`

Legacy/manual scan scripts:

- `jobs/submit_minimize_fullmodel_radius_sweep_512_uvip.sbatch`
- `jobs/submit_minimize_caseb_elastic_radius_sweep.sbatch`
- `jobs/submit_rotate_oblate_vs_sphere_radius_sweep.sbatch`
- `jobs/submit_minimize_fullmodel_direction_scan_*.sbatch`
- `jobs/submit_minimize_fullmodel_dirscan_*.sbatch`

### Minimization submission

- `jobs/submit_cnt_guide_serial.sbatch`
- `jobs/submit_continue_minimize_guide_serial.sbatch`
- `jobs/run_minimize_init_cases_local.sh`
- `jobs/run_minimize_caseb_elastic_init_cases_local.sh`
- `jobs/run_minimize_rotated_oblate_local.sh`
- `jobs/template_submit_slurm_loop.sbatch`
- `jobs/template_run_local_loop.sh`

### Scripts connecting scan to minimization / reuse

- `tools/analysis/setup_cnt_workflow.py`
- `jobs/submit_cnt_guide_serial.sbatch`
- `tools/analysis/generate_cnt_geometry_summaries.py`
- `tools/analysis/summarize_cnt_scan_from_guide.py`
- `tools/analysis/prepare_continue_dynamic_guide.py`
- `jobs/submit_continue_dynamic_guide_serial.sbatch`
- `jobs/submit_continue_minimize_guide_serial.sbatch`
- `tools/analysis/summarize_continue_from_guide.py`
- `tools/analysis/summarize_continue_minimize_from_guide.py`
- `analysis/compute_explicit_nucleation_rates.py`
- `nucleus_selector.py`

## Final Classification

The project has a **guide-driven nucleus prediction to minimization batch pipeline**, not a fully autonomous `prediction -> submit -> parse -> reuse` closed loop.
