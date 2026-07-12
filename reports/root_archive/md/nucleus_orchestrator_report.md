# Nucleus Orchestrator Report

## Installed Component

`nucleus_orchestrator.py` has been added as the top-level autonomous control layer.

It coordinates:

```text
physics prediction
  -> scan_cases.json
  -> minimization job submission
  -> job monitoring
  -> parsed_nucleus_results.csv
  -> nucleus catalog update
  -> nucleus_selector.py
  -> cuda_launch_config.json
  -> optional CUDA launch
```

## Answers

1. Is full pipeline autonomous?

   Architecturally yes. The new `FULL_AUTONOMOUS` mode is implemented in `NucleusOrchestrator.run_nucleation_pipeline()`. A single call can generate scan cases, submit jobs, monitor outputs, parse results, update the catalog, run selector, and prepare the CUDA launch config.

   Operationally, full physical execution still requires `--no-dry-run`, a working `main_cuda` binary/GPU environment, and CUDA-compatible insertion templates for the selected nucleus.

2. Is any manual intervention required?

   Not for orchestration logic. Manual intervention is only required for environment provisioning or missing data products:

   - CUDA executable / GPU runtime availability.
   - Slurm availability if `--backend slurm` is used.
   - `faceted_family_profiles.csv` generation for selected nuclei if scheduled insertion needs faceted profile templates.

3. Is catalog self-updating?

   Yes. `update_nucleus_catalog()` appends parsed results, deduplicates by `(T, xB, strain, shape_type)`, and keeps the lower-energy or higher-confidence entry. In dry-run mode it writes a preview catalog under the orchestrator run directory; in non-dry-run mode it updates root `nucleus_catalog.json`.

4. Is CUDA fully driven by physics prediction?

   The control path is now physics-driven: selector output is written into `cuda_launch_config.json`, and the config does not use manual `--scheduled-nuc-profile-dir` or `--scheduled-nuc-source-dyn-dir`.

   Current closed-loop execution is still blocked when the selected catalog entry has no CUDA-compatible `profile_dir`. In that case the orchestrator falls back to the last valid catalog entry if one exists, logs the failure, and continues without a silent crash.

5. What is the remaining gap to full self-consistency?

   The remaining gap is automatic generation of CUDA insertion templates from minimized nuclei:

   ```text
   minimized phi/xB field
     -> geometry/profile extraction
     -> faceted_family_profiles.csv
     -> source_dyn_dir/profile_dir catalog fields
     -> selector returns cuda_now_uses_predicted_nucleus=true
   ```

## New Files / Outputs

- `nucleus_orchestrator.py`
- `nucleus_orchestrator_report.md`
- `Results/orchestrator/<workflow>/input/scan_cases.json`
- `Results/orchestrator/<workflow>/jobs/minimization_jobs.json`
- `Results/orchestrator/<workflow>/parsed/parsed_nucleus_results.csv`
- `Results/orchestrator/<workflow>/selected_nucleus.json`
- `Results/orchestrator/<workflow>/cuda_launch_config.json`
- `Results/orchestrator/<workflow>/orchestrator_state.json`

## Implemented Functions

- `run_nucleation_pipeline(T, xB, strain)`
- `NucleusOrchestrator.generate_nucleus_scan_range()`
- `NucleusOrchestrator.submit_minimization_jobs(scan_cases)`
- `NucleusOrchestrator.monitor_jobs(jobs)`
- `NucleusOrchestrator.parse_results(scan_cases)`
- `NucleusOrchestrator.update_nucleus_catalog(parsed_rows)`
- `NucleusOrchestrator.select_nucleus()`
- `NucleusOrchestrator.generate_cuda_launch_config(selected_payload)`
- `NucleusOrchestrator.run_cuda_simulation(config)`

## Validation Performed

Dry-run command:

```bash
python3 nucleus_orchestrator.py \
  --T 400 \
  --xB 0.05 \
  --strain -0.01 \
  --strain-mode eyy \
  --dry-run \
  --steps 2 \
  --grid 16,16,16 \
  --delta-nm 0.1 \
  --step-nm 0.1
```

Result:

```text
nucleation_orchestrator_installed
full_pipeline_mode_active
autonomous_status = true
catalog_self_learning = true
cuda_closed_loop = false
next_upgrade_recommendation = generate CUDA-compatible profile_dir templates for selected minimized nuclei and enable --launch-cuda in non-dry-run mode
```

`cuda_closed_loop=false` is expected in the current checkout because selected/minimized nuclei still lack CUDA insertion profile templates.
