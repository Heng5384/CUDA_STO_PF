# Legacy L/X/Q One-Step Root-Cause Replay

## Reproduction

The workstation build used CUDA 12.9.86 on RTX 5080. All cases used the same 8^3 periodic state, `dt=1e-4`, a radius-2 diffuse seed, T400 thermodynamics, and disabled elasticity, GP, RSMD, external source, and global projection. Commands are captured in `scripts/pf/run_task1_one_step_replay.sh`.

Set `PF_COMPOSITION_TRACE_ONE_STEP=1`; optional `PF_COMPOSITION_TRACE_STEP=N` selects the traced step. Trace-off produces byte-identical `phi_1.vtk`, `xB_1.vtk`, and `xBtot_1.vtk` before and after instrumentation. Diagnostic atomic reductions differ only at roundoff.

The CSV uses requested semantic stage names but preserves actual order: `S0,S1,S5,S2,S3,S4,S6,S7,S8,S9`, because the executable updates phase before composition transport.

## L: lagged RHS

The first nonconservative operator is **S5 phase update**. `mass(C)` changes from `45.13246493305910` to `43.51114259520244` before mu or flux is evaluated. The signed phase-storage request is `-1.587696459484053`, but L has not performed the exact local update `q_new=q_old-delta_h*v_B`.

The L Y-RHS then uses nonlinear `dh/dt`, logistic storage, stabilizer add/subtract, and lagged `dYdt_prev` (`cuda_kernels.cu:2118-2198`, called at `main_cuda.cu:29425-29443`). At S4 it changes mass further to `43.39901357532152`; final relative loss is `-3.840808e-2`. Divergence sums to `-2.26e-16`, and projection is disabled, so neither periodic transport nor projection is the first cause.

**Classification:** first cause = phase/composition operator splitting without exact local storage transaction; secondary cause = lagged nonlinear Y inversion does not exactly compensate that transfer.

## X: x transport/projection split

X has the same first loss at **S5**, from `45.13246493305910` to `43.51114259520244`. Its update evaluates `x_t=divJ/(1-h_new)` and protects small matrix support (`cuda_kernels.cu:2762-2790`), but omits the same-step phase storage transfer. S4 ends at `43.46218372746601`; final relative loss is `-3.700842e-2`.

**Classification:** first cause = phase update at fixed matrix composition rather than fixed total storage; secondary cause = singular/protected `(1-h_new)` inversion can only transport matrix composition and cannot repair phase storage.

## Q: local matrix-storage transaction

At S5, before the Q transaction, old `q` combined with new `h` yields an infeasible intermediate. The first reported cell is `(3,4,4)`: `h=1`, `q=1.2107684437e-10`, `C=1.000000000121077`, while admissible `Cmin=Cmax=1`. The exact Q transaction detects 30 infeasible phase requests and executes `policy=rollback_entire_phi_substep` (`main_cuda.cu:28898-28932`; `cuda_kernels.cu:2826-2859`).

After rollback, S2 mass is restored exactly. Q transport reports seven bounded transport cells, resolves them by conservative local redistribution, and finishes with `mass(C)=45.13246493305918`, relative change approximately `1.9e-15` and no NaN/Inf.

**Classification:** first rejected operator = S5 phase proposal exceeding local q capacity. Unlike L/X, Q does not accept the invalid phase state; its rollback prevents mass loss. This is an operator feasibility rejection, not a successful phase advance.

## Common findings

- `mu` and flux are downstream of the first L/X failure.
- Spectral periodic divergence has roundoff-zero global sum; its `k=0` mode is structurally zero.
- No tested mode uses projection in this replay; S8 increments are zero.
- History commits only at S9, after the accepted composition state.
- No NaN/Inf occurs in this one-step case.
- The production thermodynamic closure remains **BLOCKED** independently because legacy pure-beta mobility is finite while beta has no composition degree of freedom.

Task 1 diagnoses existing modes only. It does not validate or introduce a replacement architecture.
