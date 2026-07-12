# PF Change Plan After Task 1

Task 1 changes behavior only by adding opt-in tracing. No physical equation is changed.

## Phase-1 replay instrumentation

- `PF_COMPOSITION_TRACE_ONE_STEP` parsing and deterministic CSV lifecycle are at `main_cuda.cu:24779-24807`.
- Read-only host summaries for `phi/Y/x/q/C/mu/divJ` are at `main_cuda.cu:22073-22270`.
- Trace probes cover accepted-state save (`main_cuda.cu:27425-27435`), phase completion (`main_cuda.cu:28550-28555`), Q local transfer (`main_cuda.cu:28898-28932`), mu/flux/divergence (`main_cuda.cu:28943-29118`), mode update (`main_cuda.cu:29315-29530`), projection (`main_cuda.cu:29548-29831`), and history commit (`main_cuda.cu:30207-30217`).
- Preserve trace-off execution: no changed launch arguments, no state writes, and no altered synchronization except inside the opt-in branch.

## Later gated work

- A future authoritative `C_B_tot` restart and accepted/work transaction belongs after Task 1; current legacy restart is VTK/raw `phi/x/Y` context (`main_cuda.cu:22332-22357,22434-22441`).
- A future finite-volume transport must replace, not silently reinterpret, the spectral flux chain at `main_cuda.cu:29011-29112`.
- Pure-beta mobility closure must be resolved before production approval: `D_mix` at `thermo_utils.h:546-549` and `compute_flux_single_component_kernel` at `cuda_kernels.cu:1917-1945` currently retain `D_compound` without a beta composition degree of freedom.
- RSMD, GP release/growth, inventory injection, and S3 remain outside this branch phase.
