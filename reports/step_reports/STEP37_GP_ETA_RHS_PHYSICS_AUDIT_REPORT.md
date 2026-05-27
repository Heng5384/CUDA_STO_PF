# Step 37 GP eta RHS Physics Audit

## Runtime configuration
- `gp_W_eta = 0.0`
- `gp_kappa_eta = 0.0`
- `gp_L_eta = 0.0001`
- `gp_elastic_enabled = 1`
- `gp_elastic_active_eta = 1`
- `gp_elastic_derivative_scale = 1.0`
- `gp_eps_iso = 0.0`
- `gp_obs_iface_width_nm = 0.2`
- `mu_GP0_mechanical_mixture = -153529.2328 J/mol`
- `minus_Delta_mu_r_GP(xB=0.03) = 2934.3703 J/mol`
- `matrix diffusion thermodynamics branch = raw_regular_solution`

## Code audit
- `gp_W_eta` enters `compute_eta_rhs_kernel()` as `+ gp_W_eta * g_prime_of_phi(eta)` in [cuda_kernels.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:851).
- `gp_kappa_eta` does not enter the explicit RHS; it enters the semi-implicit denominator in `eta_semi_implicit_update_kernel()` via `1 + L_eta*dt*kappa_eta*k^2` in [cuda_kernels.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:950).
- The gradient contribution written to VTK is reconstructed as `-gp_kappa_eta * laplacian(eta)` in [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:13456).
- Elastic `dgel/deta` is active when `gp_elastic_enabled && gp_elastic_active_eta`, through `compute_dgel_deta_gp_point()` in [cuda_kernels.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:280) and the scaled term added at [cuda_kernels.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:833).
- The eta update is semi-implicit in the gradient term and explicit in chemical / double-well / elastic terms.

## Interface term magnitudes (0.1 < eta < 0.9)

### step 1000
- `chem_rms = 1.297349e+00`
- `dw_rms = 0.000000e+00`
- `grad_rms = 0.000000e+00`
- `elastic_rms = 0.000000e+00`
- `full_rms = 1.297349e+00`
- `ratio_dw_over_chem = 0.000000e+00`
- `ratio_grad_over_chem = 0.000000e+00`
- `ratio_elastic_over_chem = 0.000000e+00`
- `ratio_nonchemical_over_chem = 0.000000e+00`
- `minus_Delta_mu_r_GP(eta≈0.5)_mean = 6.699278e+02`
- `h_prime(eta≈0.5)_mean = 1.863238e+00`
- `g_prime(eta≈0.5)_mean = -2.229854e-03`
- `eta_rhs_chem(eta≈0.5)_mean = -1.472320e+00`
- `eta_rhs_dw(eta≈0.5)_mean = 0.000000e+00`
- `eta_rhs_grad(eta≈0.5)_mean = 0.000000e+00`
- `eta_rhs_elastic(eta≈0.5)_mean = 0.000000e+00`

### step 2000
- `chem_rms = 3.171757e-01`
- `dw_rms = 0.000000e+00`
- `grad_rms = 0.000000e+00`
- `elastic_rms = 0.000000e+00`
- `full_rms = 3.171757e-01`
- `ratio_dw_over_chem = 0.000000e+00`
- `ratio_grad_over_chem = 0.000000e+00`
- `ratio_elastic_over_chem = 0.000000e+00`
- `ratio_nonchemical_over_chem = 0.000000e+00`
- `minus_Delta_mu_r_GP(eta≈0.5)_mean = 6.063151e+02`
- `h_prime(eta≈0.5)_mean = 1.863581e+00`
- `g_prime(eta≈0.5)_mean = -1.898781e-03`
- `eta_rhs_chem(eta≈0.5)_mean = -3.343076e-01`
- `eta_rhs_dw(eta≈0.5)_mean = 0.000000e+00`
- `eta_rhs_grad(eta≈0.5)_mean = 0.000000e+00`
- `eta_rhs_elastic(eta≈0.5)_mean = 0.000000e+00`

## Health metrics
- step 0: `eta_max=9.999500e-01`, `V_h=4.188791e+00 nm^3`, `R_eff_h=1.000000e+00 nm`, `xB_range=[2.704000e-02, 3.000000e-02]`, `drift=0.000000e+00`, `max_abs(dt*divJ)=nan`
- step 1000: `eta_max=9.999500e-01`, `V_h=7.244546e+00 nm^3`, `R_eff_h=1.200349e+00 nm`, `xB_range=[2.140000e-03, 1.211000e-02]`, `drift=2.980201e-02`, `max_abs(dt*divJ)=0.00077791762551`
- step 2000: `eta_max=9.997500e-01`, `V_h=7.248264e+00 nm^3`, `R_eff_h=1.200554e+00 nm`, `xB_range=[1.610000e-03, 7.480000e-03]`, `drift=5.435900e-04`, `max_abs(dt*divJ)=1.0417649234e-05`

## Answers
1. The current eta equation is chemical-drive dominated rather than a full Allen-Cahn GP phase-field equation with active interfacial/elastic limiting terms.
2. `gp_W_eta` and `gp_kappa_eta` are zero at runtime.
3. Elastic `dgel/deta` is active but negligible compared with chemical drive.
4. The current model does not contain an effective finite-size limiting mechanism in the eta RHS for this run.
5. Step36 runaway/continued growth is therefore not proof that the mechanical-mixture `mu_GP0` is physically wrong; it is also consistent with missing or too-weak `W_eta / kappa_eta / elastic` limiting physics.
6. The next required calibration target is all three nonchemical controls, with immediate priority on `W_eta` and `kappa_eta`, then elastic strength (`gp_eps_iso`, `gp_elastic_derivative_scale`) once the interfacial terms are nonzero.
