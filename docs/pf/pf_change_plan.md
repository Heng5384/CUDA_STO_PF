# PF Change Plan After Task 1

Task 1 changes behavior only by adding opt-in tracing. No physical equation is changed.

## Phase-1 replay instrumentation

- Add `PF_COMPOSITION_TRACE_ONE_STEP` parsing and deterministic CSV lifecycle near the runtime output setup in `main_cuda.cu:24373-24690`.
- Add read-only host summaries for `phi/Y/x/q/C/mu/divJ` near existing reduction helpers in `main_cuda.cu:3887-4041` or an adjacent diagnostics section.
- Insert trace probes at accepted-state save (`main_cuda.cu:27187-27190`), phase completion (`main_cuda.cu:28262-28304`), Q local transfer (`main_cuda.cu:28647-28680`), mu/flux/divergence (`main_cuda.cu:28698-28851`), mode update (`main_cuda.cu:29052-29245`), projection (`main_cuda.cu:29380-29524`), and history commit (`main_cuda.cu:29913-29919`).
- Preserve trace-off execution: no changed launch arguments, no state writes, and no altered synchronization except inside the opt-in branch.

## Later gated work

- A future authoritative `C_B_tot` restart and accepted/work transaction belongs after Task 1; current legacy restart is VTK/raw `phi/x/Y` context (`main_cuda.cu:22125-22149,22226-22233`).
- A future finite-volume transport must replace, not silently reinterpret, the spectral flux chain at `main_cuda.cu:28755-28851`.
- Pure-beta mobility closure must be resolved before production approval: `D_mix` at `thermo_utils.h:546-549` and `compute_flux_single_component_kernel` at `cuda_kernels.cu:1917-1945` currently retain `D_compound` without a beta composition degree of freedom.
- RSMD, GP release/growth, inventory injection, and S3 remain outside this branch phase.
