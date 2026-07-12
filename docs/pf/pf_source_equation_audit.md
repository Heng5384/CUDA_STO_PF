# PF Source Equation Audit

## Scope and build inventory

- Repository: `/Users/heng/Documents/GitHub/CUDA_STO_PF`
- Audit branch: `codex/pf-ctot-production`; starting commit: `d8e836566829eb3458641346cdaca6a2ddf3ed60`.
- Workstation: `fuxin`, RTX 5080 (16,303 MiB), driver 580.95.05, CUDA 12.9.86, GCC 13.3.0.
- Build: `make clean && make main_cuda NVCC=/usr/local/cuda-12.9/bin/nvcc CUDA_ROOT=/usr/local/cuda-12.9 -j1`.
- Build definition: `Makefile:2-13,40-42`; CUDA double-complex FFT and `double` PF fields, with elastic FFT buffers in `cufftComplex`/`float` (`main_cuda.cu:26626-26814`).
- Deterministic seed is `PFParams::seed` (`pf_params.h:18`) and seed placement calls `srand(seed)` (`main_cuda.cu:2318-2336`).
- Runtime snapshots are parameter text plus VTK/raw field files; continuation reads VTK/raw fields and an adjacent `pf_input.params` (`main_cuda.cu:22125-22149,22226-22233`). There is no versioned authoritative `C_B_tot` restart in the legacy path.

Pre-instrument workstation gate: the clean CUDA build passed. The existing host storage test and the new directional derivative test passed. A one-step 8^3 PF-only L baseline (`dt=1e-4`, radius 2 grid units, projection/GP/RSMD/elasticity off) completed without NaN, but changed mean total storage from `8.81493456e-02` to `8.47636984e-02` (`-3.840808e-02` relative). This is a regression observation, not a production pass.

## Executable continuous expressions

The interpolation is exactly

`h(phi)=phi^3(6phi^2-15phi+10)`, `h'=30phi^2(1-phi)^2`, and `g=phi^2(1-phi)^2`; source: `phase_functions.h:6-17,50-57`.

The matrix molar free energy is the regular-solution CALPHAD form whose partial chemical potentials are

`mu_A=G_PbTe+RT ln(1-x)+Lx^2` and `mu_B=G_Ag2Te+RT ln(x)+L(1-x)^2`, with `L=41212.9-18.05T`; source: `thermo_utils.h:338-375,356-359`. Above `x=0.09`, optional quadratic chemical-potential extrapolation is applied (`thermo_utils.h:395-430`).

The local density conversion and mixed molar energy are

`c=1/[Vm_alpha(x)(1-h)+Vm_compound*h]` and `mu_total=(1-h)[(1-x)mu_A+x mu_B]+h mu0_compound`; source: `thermo_utils.h:490-510`.

The executable composition potential is

`mu_C=c[mu_B-mu_A-c*mu_total*dVm_alpha_dxB] - eps_iso_over_vB*tr(sigma)`

from `cuda_kernels.cu:1199-1256`. No composition-gradient term is present in this kernel. The elastic term is nonlocal through the FFT elasticity solve and local hydrostatic derivative.

The phase explicit derivative is

`f_phi=W*g'(phi)+c*h'(phi)[delta_mu-c*mu_total*volume_term]+dgel/dphi`,

where `delta_mu=mu0-v_A mu_A-v_B mu_B-elastic_shift` and `volume_term=Vm_compound-Vm_alpha+dVm_dx*(x-v_B)`; source: `cuda_kernels.cu:390-464,466-551`. The gradient term is treated semi-implicitly as `-kappa_phi laplacian(phi)` by the Fourier update (`main_cuda.cu:28262-28271`). Thus the chemical phase derivative is intended at fixed total storage, not fixed matrix `x`.

For current production inputs (`v_A=0`, `v_B=1`, `Vm_alpha=Vm_beta=1`, `dVm/dx=0`), this reduces exactly to `h'(mu0-mu_B)`, the fixed-`C` derivative of `(1-h)f_alpha(x)+h mu0` with `C=(1-h)x+h`.

The transport coefficient is `M_eff=stabilized_meff(D_mix,Gamma)` with `D_mix=(1-h)D_alpha+hD_compound` (`thermo_utils.h:546-549`) and `Gamma=c*d(mu_B-mu_A)/dx`, finite-differenced at `dx=1e-5` and capped at `x=0.08` when convex stabilization is enabled (`thermo_utils.h:551-577`). Flux is `J=M_eff grad(mu)` (`cuda_kernels.cu:1917-1945`) and the source-free equation uses `C_t=div(J)` under the code's sign convention.

The phase kinetics are Allen-Cahn semi-implicit in Fourier space, `phi_k^{n+1}=[phi_k^n-dt L_phi f_exp,k]/[1+dt L_phi kappa k^2]` (launcher at `main_cuda.cu:28262-28271`; exact kernel in `cuda_kernels.cu`). Periodicity follows wrapped spectral wave numbers and cuFFT transforms (`cuda_kernels.cu:2101-2112`). Every inverse FFT is explicitly normalized by `invN`, e.g. gradient/divergence at `main_cuda.cu:28780-28781,28844-28851` and phase clamp at `main_cuda.cu:28271-28304`.

## Production closure finding

**BLOCKED:** the audited T400 physical parameter snapshot has `D_compound=0.09`, while beta is fixed at `v_B=1` and has no independent beta composition/susceptibility state. Therefore `D_mix -> D_compound` and legacy `M_eff` remains finite in a pure-beta cell although there is no beta composition degree of freedom. This is the exact inconsistency required by the production rules; no support threshold resolves it. Task 1 records it and does not change it.
