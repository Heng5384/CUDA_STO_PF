# Phi/Eta Iteration Consistency Audit

Scope: read-only audit of RHS composition, semi-implicit updates, Fourier handling, and asymmetries.

## Bottom line

- Core `phi` / `eta` update algebra is symmetric.
- The main asymmetry is intentional physics plumbing, not a missing denominator term.
- High-risk asymmetry count: **0**
- Medium-risk asymmetry count: **3**
- Low-risk asymmetry count: **4**

## (a) RHS composition

- `phi_rhs` has explicit double-well `W * g'(phi)` at [cuda_kernels.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:432), plus bulk chemical driving at [cuda_kernels.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:436) and elastic driving at [cuda_kernels.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:453). The double-well is not absorbed into the chemical potential path; it is added explicitly.
- `eta_rhs` has the same structural split: bulk/chemical drive at [cuda_kernels.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:843), explicit double-well `gp_W_eta * g'(eta)` at [cuda_kernels.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:865), and elastic drive at [cuda_kernels.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:846).
- The `compute_eta_rhs_components_kernel` mirror makes that split explicit as `chem_part`, `dw_part`, and `elastic_part` at [cuda_kernels.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:868-963) in the source file.

## (b) Semi-implicit updates

- `phi_semi_implicit_update_kernel` is at [cuda_kernels.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:741).
- `eta_semi_implicit_update_kernel` is at [cuda_kernels.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:965).
- The algebra is the same in both kernels: `denom = 1.0 + L_dt * kappa * k2[idx]`, then `(old - L_dt * rhs) / denom`. That is a literal structural match, not just conceptually similar.

## (c) Fourier handling

- Both updates use the same `KS.d_k2` array in the driver: `phi` at [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:15882-15907), `eta` at [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:16374-16386) and again at [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:16898-16906).
- `phi` normalization/clamp is looser: [-1e-6, 1+1e-6] in [cuda_kernels.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:991-1003).
- `eta` normalization/clamp is stricter: hard [0,1] in [cuda_kernels.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:1006-1017).

## (d) Stability indicator

- The startup summary computes `S_phi_total`, `S_eta_total`, and `CFL_diff` but only prints them; it does not assert them at [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:6568-6597).
- Suggested audit thresholds:
  - `> 2`: warn
  - `> 4`: fatal
- Rationale: the semi-implicit denominator damps the linear diffusion part, but the explicit source terms are still carried step-to-step. Once the combined step indicator is above ~2, pseudo-oscillatory behavior and precision loss become plausible; above ~4, the step is usually too stiff for comfortable production use.

## (e) Asymmetry list

- Phi-only minimize / projection helpers: `add_volume_constraint_kernel` and `apply_volume_projection_kernel` at [cuda_kernels.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:559-589), plus the residual helper at [cuda_kernels.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:630-644) and the post-projection driver at [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:15894-15907). Risk: **medium**. This is intended because `phi` carries the constrained phase fraction loop.
- Eta-only GP feature paths: `gp_to_beta` feasibility / event logic around [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:3785-4038) and the raw-reaction / observed-GP init plumbing around [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:7896-8134), [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:8568-8585), [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:14273-14344), [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:16863-16906). Risk: **medium**. Physics-specific by design, but the path is much richer than phi’s.
- Y-only coupling / Picard toggles: explicit previous-time-level, gamma-term suppression, and Picard controls at [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:14737-14829) and the kernel-side toggles at [cuda_kernels.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:2107-2240). Risk: **low** for phi/eta symmetry, because these are orthogonal transport numerics, not order-parameter PDE structure.
- Normalization asymmetry: phi’s small overshoot tolerance vs eta’s hard clamp at [cuda_kernels.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:991-1017). Risk: **low**.

## (f) Verdict

The code is iteration-consistent at the solver level. What remains is mostly intentional feature asymmetry and diagnostic layering.

