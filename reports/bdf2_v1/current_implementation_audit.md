
# Current IMEX-BDF2 Implementation Audit

Audit date: 2026-07-16. Scope: workstation-only PF T400 fixed-step candidate.

## Verdict

`BDF2_implementation_status=PASS_FIXED_STEP_IMEX_BDF2_V1_IMPLEMENTED`

The default-off selector `ctot_jichen_imex_bdf2_v1` is now a complete
transport-first IMEX method. It performs one fixed-`phi_E` conservative
transport solve, one fixed-final-`Ctot` phase PDAS solve, one final mechanics
solve, and one cold audit. It does not execute final transport polish, outer
M3/Anderson, or a second phase solve.

## Final Implementation Map

| Contract | Final source location | Status |
|---|---|---|
| selector and version strings | `main_cuda.cu:1207-1229` | complete, default off |
| parameter validation | `main_cuda.cu:1975-2067` | complete |
| BDF2 context/rate/work kernels | `main_cuda.cu:3176-3254` | complete |
| checkpoint metadata parser | `main_cuda.cu:4741-4904` | complete |
| strict history loader | `main_cuda.cu:29411-29515` | complete |
| accepted-history ownership | `main_cuda.cu:29944-30566` | complete |
| BE/BDF2 selector and fallback reason | `main_cuda.cu:33080-33158` | complete |
| exact/ULP storage active set | `main_cuda.cu:33653-33862` | complete |
| fixed-final-C phase PDAS | `main_cuda.cu:33956-34500` | complete |
| BDF2 mass identity | `main_cuda.cu:35351-35375` | complete |
| endpoint discrete-work audit | `main_cuda.cu:37387-37618` | complete |
| atomic history commit | `main_cuda.cu:38383-38422` | complete |
| history raw files and metadata | `main_cuda.cu:38495-38582` | complete |

Source hashes: `{"ctot_transport_bound_utils.h": "ba0cf736165de65610f8e58b7eaa34468dbf81a7be2f955f1148b280c183d237", "cuda_kernels.cu": "a4d269067ab2de6059af1d48c333eda80aec43268c3f61036eab1a66e760b81a", "cuda_kernels.h": "3eb5cfa9ed35ddb533f43d5847ca270a0ac3a4ddb389770140e7fa7e7c425b53", "main_cuda.cu": "08c87c390be1b696bde8cebb3f2bd0bc70b64841464038a382507f00a81ace7e", "pf_params.h": "b42e6f9c83febf727b55dc12102572d08e1b970b36cbe7f0f0ebcbf19d01c778"}`.

## Preserved Routes

- Lie-BE v2 remains selectable and passed the current-binary comparator run.
- Failed staggered-v1 remains present as a default-off negative control.
- Legacy/P2 one-step replay is bitwise equal for `phi`, `xB_alpha`, and `Ctot`.
- Thermodynamics, mobility, `Dalpha`, `Dbeta`, `Lphi`, `gamma`, `lambda`,
  `dx`, and `h(phi)` were not changed.
