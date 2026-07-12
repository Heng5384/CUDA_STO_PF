# Root Cleanliness Final Check

## Root Files

### All root files
```text
./.DS_Store
./.gitignore
./MIGRATION_NOTES.md
./Makefile
./README.md
./Unit_Psedobinary.py
./analyze_nucleation_event_observables.py
./cnt_scan_pipeline.py
./coupling_interface.json
./cuda_common.cu
./cuda_common.h
./cuda_kernels.cu
./cuda_kernels.h
./io_vtk_cuda.h
./main_cuda.cu
./normalize_nucleation_physical_units.py
./nucleus_catalog.json
./nucleus_catalog.schema.json
./nucleus_geometry_mapper.py
./nucleus_orchestrator.py
./nucleus_selector.py
./pf_dynamics_core.py
./pf_params.h
./phase_functions.h
./physical_inputs.example.json
./physical_units_config.json
./plot_step41D_gp_beta_driving_forces.py
./run_step38d_gp_L_eta_reference_scan.sh
./run_step38e_phi_eta_step_delta_diag.sh
./run_step38f_phi_eta_rhs_attribution_diag.sh
./run_step38j_eta_S218b_Mratio_scan.sh
./s_field_definition.py
./selected_nucleus.json
./test_memory_ledger.cu
./thermo_utils.h
./validate_minimal_physics_loop.py
./validate_nucleation_theory_vs_cuda.py
```

### Root markdown / csv / text / log / out / err
```text
./MIGRATION_NOTES.md
./README.md
```

## Status

- Root report outputs remaining: none.
- Root retains only entry docs (`README.md`, `MIGRATION_NOTES.md`) plus source/scripts/directories that are not report artifacts.
- No new script in this audit is writing report outputs back to root as a primary destination.

## Root Directories

```text
.
./.claude
./.git
./CNT_SCAN_WORKSTATION_RUN
./Results
./Results_scan
./__pycache__
./analysis
./archive
./configs
./jobs
./nucleus_generator
./nucleus_parametric_generator
./outputs
./reports
./scripts
./shape_library
./tools
```
