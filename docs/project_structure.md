# Project Structure

## Runtime and build

- `main_cuda.cu`: executable entry point, parameter validation, initialization, time integration, ledgers, and output.
- `cuda_kernels.cu`, `cuda_kernels.h`: CUDA kernels and launch interfaces.
- `cuda_common.cu`, `cuda_common.h`, `thermo_utils.h`, `phase_functions.h`, `io_vtk_cuda.h`: shared CUDA, thermodynamic, phase, and VTK support.
- `pf_params.h`: runtime parameter schema.
- `Makefile`: `main_cuda` build entry; see the root README for CUDA path and architecture settings.

## Reproducible inputs

- `physical_inputs.example.json` and `Unit_Psedobinary.py`: physical-input to code-unit conversion.
- `params/`: accepted, diagnostic, and regression parameter sets. Mode L/X/Q and PF-only comparison inputs remain here even when a mode fails.
- `data/`, `shape_library/`, and nucleus-library assets: physical tables, accepted seeds, profiles, and runtime libraries.
- `configs/`: compact workflow configuration.

## Workflows and evidence

- `jobs/`: local and cluster launchers plus site environment handling.
- `scripts/`: validation, preparation, regression, and small smoke workflows.
- `tools/analysis/` and `analysis/`: analysis and report-generation code.
- `reports/`: compact scientific reports, acceptance evidence, selected tables, and figures.
- `reports/step_reports/`: historical STEP reports retained for provenance.
- `reports/root_archive/`: reports and tables moved out of the repository root by the cleanup pass.

## Generated or local-only material

- `Results/`, `Results_scan/`, `outputs/`: simulation output trees, retained locally and ignored by Git.
- `build/`, `cmake-build-*`, `main_cuda`, object/device binaries: reproducible build products, ignored by Git.
- `tmp/`, `tmp_codex_ops/`, `tmp_runtime_4grid_*`, `output/`: scratch or rendered artifacts, ignored by Git. Existing `tmp_codex_ops/` data is retained pending explicit scientific review.
- VTK and RAW field dumps are ignored; compact CSV/Markdown summaries should carry acceptance conclusions and provenance.

## Cleanup policy

Scientific relevance takes priority over directory names. Failed Mode L/X/Q reports, capacity-bound failures, projection audits, S3 source-only validation, GP ledgers, and handoff evidence are retained when they explain an acceptance gate or rejected route. Large raw data is not deleted solely because a summary exists unless reproducibility and supersession are explicitly confirmed.
