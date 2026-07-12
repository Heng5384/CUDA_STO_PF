# Nucleus Minimization Pipeline Map

## Verdict

There is a guide-driven automatic pipeline from predicted critical radius to a batch of CUDA minimization runs:

```text
physical_inputs + T + xB + strain modes
  -> Schur/CNT rc prediction
  -> guide_cnt_scan.csv radius candidates
  -> Slurm serial guide runner
  -> ./main_cuda --mode=minimize --radius-phys-nm <candidate>
  -> energy_minimize_*.csv + VTK
  -> summary.txt geometry + current_results_master_table_fitted.csv
  -> nucleation_rate_table.csv
```

It is not a single closed autonomous submit-and-reuse daemon. The workflow is automatic once each stage is invoked, but stage transitions are normally manual commands or `sbatch` invocations.

## A. Nucleus Candidate Generation

| stage | file | function / line range | input data type | output data type | automatic trigger? |
|---|---|---:|---|---|---|
| strain/load case enumeration | `tools/analysis/compute_schur_rc_predictions.py` | `build_cases`, lines 93-103 | `strains: list[float]`, `modes: list[str]` | `list[LoadCase]` with mode, strain, fixed/free strain indices, external strain tensor | Automatic inside prediction/workflow scripts |
| Schur elastic and rc prediction | `tools/analysis/compute_schur_rc_predictions.py` | `calc_elastic_energy_schur`, lines 166-197; `main`, lines 272-299 | physical input JSON, `LoadCase`, CALPHAD driving force | rows with `rc_schur_nm`, `DG_chem_Jm3`, `E_el_Jm3`, `DG_net_Jm3` | Manual script invocation |
| standardized CNT workflow generation | `tools/analysis/setup_cnt_workflow.py` | `_scan_radii`, lines 48-50; `main`, lines 53-322 | CLI args, physical input JSON, Schur predicted `rc_nm` | `input/schur_rc_predictions.csv`, `input/guide_cnt_scan.csv`, `input/workflow_meta.json` | Manual script invocation |
| radius candidate set | `tools/analysis/setup_cnt_workflow.py` | lines 221-281 | `rc_schur_nm`, `window_nm`, `step_nm` | `guide_cnt_scan.csv` rows with `row_type=scan_point`, `radius_nm`, `case_tag`, output paths | Automatic within `setup_cnt_workflow.py` |
| path and output naming | `tools/analysis/workflow_utils.py` | `case_output_rel`, lines 102-138; `reference_case_output_rel`, lines 141-178 | candidate radius, case tag, grid, dt, xB, elastic flag | expected `case_dir_rel`, `energy_csv_rel`, `phi_final_rel`, `pf_input_rel` | Automatic within guide generation |

## B. Energy Minimization Trigger

| stage | file | function / line range | input data type | output data type | automatic trigger? |
|---|---|---:|---|---|---|
| guide row reader | `jobs/submit_cnt_guide_serial.sbatch` | lines 46-64 | `GUIDE_CSV=.../guide_cnt_scan.csv` | shell export block per CSV row | Automatic inside Slurm job |
| case completeness guard | `jobs/submit_cnt_guide_serial.sbatch` | `cnt_case_complete`, lines 84-93 | guide row paths | skip/run decision | Automatic inside Slurm job |
| per-row minimize command | `jobs/submit_cnt_guide_serial.sbatch` | `run_case`, lines 184-258 | guide row fields: `radius_nm`, `case_tag`, `T_C`, `xB_out`, strain components, paths | `./main_cuda ... --mode=minimize --minimize-full-model --radius-phys-nm <radius_nm>` | Automatic inside Slurm job after manual `sbatch` |
| reference matrix-only cases | `jobs/submit_cnt_guide_serial.sbatch` | lines 201-250 | `row_type=reference` guide row | raw uniform matrix init + one-step reference energy | Automatic inside Slurm job |
| continue-minimize from selected source | `jobs/submit_continue_minimize_guide_serial.sbatch` | `run_case`, lines 124-169 | `guide_continue_dynamic.csv` row with `continue_phi_vtk`, `start_radius_nm` | `./main_cuda --mode=minimize-continue ...` | Automatic inside Slurm job after manual `sbatch` |

## C. Job Submission Mechanism

| mechanism | file | line range | input data type | output data type | automatic trigger? |
|---|---|---:|---|---|---|
| Slurm batch entry for CNT scan | `jobs/submit_cnt_guide_serial.sbatch` | lines 1-27, 260-277 | environment variable `GUIDE_CSV`; Slurm resource headers | sequential execution of all guide rows | Manual `sbatch` submission |
| Slurm batch entry for continue-minimize | `jobs/submit_continue_minimize_guide_serial.sbatch` | lines 1-24, 171-188 | environment variable `GUIDE_CSV`; Slurm resource headers | sequential continue minimization runs | Manual `sbatch` submission |
| documented end-to-end commands | `tools/analysis/critical_radius_continue_workflow.md` | lines 675-724, 811-846 | user shell commands | documented command sequence from setup to submit to summarize | Manual execution |
| older/manual sweep templates | `jobs/submit_minimize_fullmodel_radius_sweep_512_uvip.sbatch` and related `jobs/submit_minimize_*sweep*.sbatch` | script-level | hard-coded radii / environment variables | direct `main_cuda` radius sweeps | Manual scripts, not tied to Schur guide by default |

## D. Output Parsing / Storage

| stage | file | function / line range | input data type | output data type | automatic trigger? |
|---|---|---:|---|---|---|
| minimization energy log | `main_cuda.cu` | open header lines 14219-14245; write rows lines 17602-17627 | runtime minimize state, energies, residuals | `energy_minimize_T...csv` with `F_total_CNT_hat`, `F_total_excess_hat`, residuals | Automatic during `main_cuda --mode=minimize` |
| final VTK fields | `main_cuda.cu` | output block around lines 17900-18005 and final output logic nearby | device fields `phi`, `xB`, diagnostics | `phi_final_*.vtk`, `xB_final_*.vtk`, diagnostics | Automatic during `main_cuda` output |
| geometry summary generation | `tools/analysis/generate_cnt_geometry_summaries.py` | `main`, lines 44-96 | `guide_cnt_scan.csv`, `phi_final_*.vtk`, `pf_input.params` | per-case `summary.txt` with axes/aspect/voxel geometry | Manual script invocation |
| CNT scan summary | `tools/analysis/summarize_cnt_scan_from_guide.py` | `main`, lines 141-347 | `guide_cnt_scan.csv`, per-case `energy_minimize_*.csv`, `summary.txt`, reference rows | `cnt_scan/current_results_master_table_fitted.csv`, `reference_energy.csv` | Manual script invocation |
| peak fitting | `tools/analysis/analyze_cnt_peak_table.py` | `fit_local_peak`, lines 24-65; `parse_summary_file`, lines 74-220 | radius-energy points, summary text | fitted `rc_cnt_fit_nm`, shape metrics | Called by summary scripts |
| continue guide selection | `tools/analysis/prepare_continue_dynamic_guide.py` | lines 47-177 | `current_results_master_table_fitted.csv`, `guide_cnt_scan.csv` | `input/guide_continue_dynamic.csv`; source radius chosen from `rc_cnt_fit_nm + margin` or fallback | Manual script invocation |
| continue summary | `tools/analysis/summarize_continue_from_guide.py` | lines 46-113 | continue guide, continue summary files | `continue_dynamic/growth_summary.csv` or `continue_minimize/minimize_summary.csv` | Manual script invocation |
| formal nucleation-rate table | `analysis/compute_explicit_nucleation_rates.py` | `scan_case_dirs`, lines 854-920; energy collection lines 1840-1870 and 1881-1920; `write_outputs`, lines 3210-3364 | workflow root with case dirs, `energy_minimize_*.csv`, `summary.txt`, reference files | `analysis/nucleation_rate/nucleation_rate_table.csv`, diagnostics, metadata, plots | Manual script invocation |

## Chain Reconstruction

```text
A. candidate generation
   setup_cnt_workflow.py
   - uses compute_schur_rc_predictions.build_cases/calc_elastic_energy_schur
   - computes rc_schur_nm
   - expands rc_schur_nm into radius_nm = rc_schur_nm +/- window_nm in step_nm increments
   - stores guide_cnt_scan.csv

B. energy minimization trigger
   jobs/submit_cnt_guide_serial.sbatch
   - reads every guide_cnt_scan.csv row
   - exports row fields into shell variables
   - launches ./main_cuda --mode=minimize --radius-phys-nm "${radius_nm}"

C. job submission
   user runs:
   GUIDE_CSV=.../guide_cnt_scan.csv sbatch jobs/submit_cnt_guide_serial.sbatch
   The Slurm script then loops automatically through the guide.

D. output parsing / storage
   main_cuda.cu writes energy_minimize_*.csv and VTK.
   generate_cnt_geometry_summaries.py writes summary.txt.
   summarize_cnt_scan_from_guide.py writes current_results_master_table_fitted.csv.
   compute_explicit_nucleation_rates.py writes nucleation_rate_table.csv.
```

## Does rc_predicted Directly Drive Minimize Job Generation?

YES, in the guide-driven sense:

- `tools/analysis/setup_cnt_workflow.py`, lines 116-127 computes `rc_nm`.
- `tools/analysis/setup_cnt_workflow.py`, lines 221-281 expands that `rc_nm` into scan rows with `radius_nm`.
- `jobs/submit_cnt_guide_serial.sbatch`, lines 221-242 passes each guide row's `radius_nm` to `main_cuda --mode=minimize --radius-phys-nm`.

NO, in the fully autonomous sense:

- `setup_cnt_workflow.py` does not call `sbatch`.
- The Slurm submission step is a separate manual command.
- Postprocessing and reuse are separate manual commands.
