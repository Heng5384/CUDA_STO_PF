# PF-only conserved-Y zero-mode implementation contract

## Scope

This implementation is selected only by:

```text
--pf-zero-mode PF_CONSERVED_Y_ZERO_MODE_V1
--pf-zero-mode-backend HOST_NEWTON_BISECTION_V1
```

Legacy behavior remains the default when `--pf-zero-mode OFF` is used.
The selected mode is fail-closed unless all GP, eta, RSMD, scheduled-source,
dynamic-bridge, projection, Picard, and legacy Y-modifier paths are disabled.

## Selected operator

The accepted composition state is

\[
C_{B,\mathrm{tot}}=(1-h(\phi))x_{B,\alpha}+h(\phi)v_B,\qquad
x_{B,\alpha}=\sigma(Y).
\]

The selected explicit context is `SM_EXPLICIT_CONTEXT_N_V1`: chemical
potential, mobility, flux, and SM coefficients use \(\phi_n\).

The selected reaction discretization is `SM_TANGENT_N_V1`:

\[
S_h =
h'(\phi_n)\frac{\phi_{n+1}-\phi_n}{\Delta t}
\left(v_B-x_{B,\alpha,n}\right).
\]

After the semi-implicit FFT update has produced the nonzero-mode field
\(Y^\star\), the solver finds one scalar \(\lambda\) from

\[
F(\lambda)=
\sum_i\left[
(1-h(\phi_{n+1,i}))\sigma(Y^\star_i+\lambda)
+h(\phi_{n+1,i})v_B
\right]-M_0=0.
\]

The backend is FP64 bracketed Newton with bisection fallback. It validates the
entire shifted field against the declared Y bounds before applying the shift.
It does not clip the accepted field, project physical mass, assemble a sparse
matrix, or perform an implicit spatial solve. A failed scalar solve restores
the saved phase/composition state and aborts the run.

## Applied changes

| File | Final lines | Change | Mass/bound semantics | Test coverage |
|---|---:|---|---|---|
| `cuda_kernels.cu` | 2201–2269 | Added the selected tangent-at-\(n\) SM reaction kernel. | Uses \(\phi_n\) for explicit coefficients and \(\phi_{n+1}-\phi_n\) only for the time derivative. | CUDA continuous/restart and three-grid timing matrix. |
| `cuda_kernels.cu` | 3324–3376 | Added unclipped FP64 sigmoid mass, derivative, bound-validation, and final-shift kernels. | The scalar is rejected if any final Y is outside the allowed interval. | Final mass audits through 192 steps on 64³, 128³, and 195³. |
| `cuda_kernels.cu` | 3583–3640, 5571–5616 | Added paired persistent reduction for mass and derivative. | Same deterministic FP64 tree for both values; one final device-to-host pair transfer and no per-iteration allocation. | Bytewise continuous/restart equality and performance matrix. |
| `main_cuda.cu` | 4070–4242 | Added parameter fingerprint and bracketed host scalar solver. | Convergence, finite derivative, bracketing, and full-field bounds are mandatory. | Host provenance test and CUDA qualification. |
| `main_cuda.cu` | 24133–24228 | Added the pure-PF runtime gate and solver provenance. | Prohibited GP/source/projection paths fail before stepping. | CUDA logs record `gp_paths_enabled=false`. |
| `main_cuda.cu` | 29187–29214, 29656–29798 | Selected \(\phi_n\) context, tangent reaction, nonzero FFT update, and zero-mode correction. | No change to the legacy OFF path. | OFF comparator plus selected-mode runs. |
| `main_cuda.cu` | 32102–32154 | Added exact checkpoint materialization. | Stores accepted `phi`, `Y`, `xB`, `dY_dt_prev`, target mass, scalar state, step, and provenance. | Step-48 restart reaches a byte-identical step-96 checkpoint. |
| `pf_zero_mode_checkpoint.h` | 10–65 | Declared checkpoint v2 state and provenance. | Time-level and reaction selectors are restart identities. | Round-trip and mismatch rejection. |
| `pf_zero_mode_checkpoint.cpp` | 13–295 | Added checked, checksummed, atomic serialization. | Corruption or provenance mismatch is rejected before fields are returned. | `PASS_PF_ZERO_MODE_CHECKPOINT_PROVENANCE_V1`. |
| `tests/test_pf_zero_mode_checkpoint.cpp` | 46–130 | Added round-trip, backend/time-level/reaction mismatch, and corruption tests. | No partial load on invalid evidence. | Built and passed on the cluster qualification node. |
| `io_vtk_cuda.h` | 9–45 | Added an explicit benchmark-only `CUDA_STO_SUPPRESS_VTK_OUTPUT=1` gate. | Default field-output behavior is unchanged; the gate changes no state or equation. | 400³/512³ audit confirmed zero VTK files and explicit suppression markers. |

## Physical-unit qualification contract

The eta production branch's converter inputs were reused without enabling eta
or GP dynamics:

```text
temperature_C = 400
PF_DX_M = 1.0e-9
PHYS_LAMBDA_SM_M = 4.0e-9
dt_code = 0.02
t_real_unit = 41.12958542455477 s
physical_dt = 0.8225917084910954 s
```

The converter audit also fixed:

```text
dx = dy = dz = 1
ic_phi_iface_w = 2
kappa_phi = 2.0000000000000004
D_alpha = 400.00000000000006
D_compound = 4
L_phi = 9.599212970350372
```

The existing seed profile and radius were not changed or tuned.
