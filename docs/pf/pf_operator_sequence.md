# PF Operator Sequence

## Legacy dynamics call order

The accepted arrays at loop entry are `d_phi_r`, `d_Y_r`, `d_xB_r`, and history `d_dY_dt_prev_r`. At `main_cuda.cu:27187-27190`, `phi^n`, `eta^n`, and `Y^n` are copied into saved buffers.

1. **Save accepted state:** `d_phi_n_saved=phi^n`, `d_Y_n_saved=Y^n` (`main_cuda.cu:27187-27190`).
2. **Phase RHS:** `compute_phi_rhs_kernel(phi^n,x^n)` (`main_cuda.cu:28198-28230`).
3. **Phase solve:** forward FFT, dealias, semi-implicit update, inverse FFT, then normalize/clamp (`main_cuda.cu:28262-28304`). The live `d_phi_r` is now `phi^{n+1}`.
4. **Q-only phase transaction:** Q applies `q <- q-[h(phi^{n+1})-h(phi^n)]v_B`; an infeasible request rolls back the entire phase substep (`main_cuda.cu:28647-28680`; kernel `cuda_kernels.cu:2826-2859`). L and X do not perform this exact local transaction.
5. **Composition context and mu:** L/X use `phi^n` only when `enable_Y_rhs_previous_time_level=1`; otherwise they use `phi^{n+1}`. Q always uses `phi^{n+1}` (`main_cuda.cu:28692-28717`). `compute_mu_x_kernel` reconstructs `x` from the current live `Y`, overwriting `d_xB_r` (`cuda_kernels.cu:1223-1236`).
6. **Spectral flux/divergence:** `mu -> FFT -> grad(mu) -> normalize -> M_eff*grad(mu) -> FFT -> ik·J -> inverse FFT -> normalize` (`main_cuda.cu:28755-28851`). The divergence buffer is explicitly zeroed before directional accumulation (`main_cuda.cu:28767`), and `k=0` is exactly zero because each contribution is multiplied by its wrapped `k_alpha` (`cuda_kernels.cu:2101-2112`).
7. **Stabilizer context:** Laplacian is calculated from `Y`, `x`, or `q`, and `mean_DY` is reduced (`main_cuda.cu:28969-28999`).
8. **Mode update:**
   - L: assemble lagged `Y` RHS, including `dh/dt`, stabilizer add/subtract, and optional `gamma*dYdt_prev`; solve in Fourier space and clamp (`main_cuda.cu:29114-29206`; `cuda_kernels.cu:2118-2198,2368-2405`).
   - X: evolve `x` with fixed-new-phi storage inversion `divJ/(1-h_new)` plus spectral stabilizer, then clamp and rebuild `Y` (`main_cuda.cu:29060-29079`; `cuda_kernels.cu:2762-2813`).
   - Q: after the exact phase transaction, explicitly update `q += dt*divJ`; bound violations invoke host local redistribution, then reconstruct `x/Y` (`main_cuda.cu:29052-29113`; `cuda_kernels.cu:2911-2935`).
9. **Projection:** optional global `Y` shift runs after the update (`main_cuda.cu:29262-29524`). It is off in the Task-1 PF-only replay.
10. **History commit:** `dYdt_prev=(Y^{n+1}-Y^n)/dt` is written only after update/projection (`main_cuda.cu:29913-29919`; kernel `cuda_kernels.cu:3331-3342`).

## Time-level and alias table

| Array | Entry | During step | Commit |
|---|---|---|---|
| `d_phi_r` | accepted `phi^n` | overwritten by normalized/clamped `phi^{n+1}` | live accepted state |
| `d_phi_n_saved` | work | immutable snapshot `phi^n` | overwritten next step |
| `d_Y_r` | accepted `Y^n` | L/X/Q work and projected state | live accepted state |
| `d_Y_n_saved` | work | immutable `Y^n` | overwritten next step |
| `d_xB_r` | derived from `Y^n` | overwritten by mu and reconstruction kernels | derived `x^{n+1}` |
| `d_xB_prev_r` | work | snapshot immediately before flux | not authoritative |
| `d_q_alpha_r` | Q authoritative matrix storage | phase transfer and transport | Q accepted storage |
| `d_divJ_r` | work | normalized physical divergence | overwritten next step |
| `d_dY_dt_prev_r` | accepted history | read by L RHS | overwritten at line 29917 |
| `d_phi_rhs_k` | aliased | phase RHS, then `d_mu_x_k` | scratch only |

There is no general legacy step-rejection transaction. Q's infeasible phase substep restores `phi` and rebuilds `q`, but a later fatal transport/projection failure returns without a full rollback object. Because history is committed last, those fatal paths do not update `dYdt_prev`; other live work arrays may already be mutated when execution aborts.
