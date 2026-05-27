# Audit: STO_SM S3 vs CUDA Y Equation

Date: 2026-05-11

Document audited: `/Users/heng/Documents/STO_SM_with_elastic_notes.docx`

Code audited:
- `cuda_kernels.cu`
- `main_cuda.cu`
- `thermo_utils.h`
- `phase_functions.h`

Scope: formula/code audit only. No new large GPU run was launched.

## 1. STO_SM S3 Formula Summary

The S3 text extracted from the docx gives the following numerical treatment.

Allen-Cahn:

```text
dξ/dt = -Lξ (fξ - κ ∇²ξ)
ξ^{n+1}-ξ^n
------------- = -Lξ fξ^n - Lξ κ k² ξ^{n+1}
    Δt
```

For the composition equation, STO_SM notes that the equation is meaningful only in matrix/interface because the prefactor `1-h(ξ)` vanishes in the compound. To avoid division by this prefactor, the prefactor defect is moved to the RHS and evaluated explicitly. A mean coefficient is also introduced for semi-implicit stabilization:

```text
∂xB/∂t = Dbar ∇² xB + f_c                         (S3.3)

f_c = divJ - (∂h/∂ξ)(∂ξ/∂t) * (vB/(vA+vB)-xB)
      - Dbar ∇² xB
      - [prefactor correction term; docx text extraction partly ambiguous]
```

Fourier/BDF1:

```text
(xB^{n+1}-xB^n)/Δt = f_c^n - Dbar k² xB^{n+1}     (S3.4)
```

The logit variable is:

```text
Y = ln[xB/(1-xB)]
dxB/dY = exp(Y)/(1+exp(Y))² = xB(1-xB)
```

The governing Y equation in S3.5 is:

```text
(1-h) * dxB/dY * ∂Y/∂t
  = divJ - (∂h/∂ξ)(∂ξ/∂t) * (vB/(vA+vB)-xB)
```

For this code's pseudo-binary setup, `v_A=0`, `v_B=1`, so `vB/(vA+vB)=1`, matching the code's use of `v_B - xB`.

The modified semi-implicit Y equation is:

```text
∂Y/∂t = Dbar_Y ∇²Y + f_Y                           (S3.6)

Dbar_Y = mean( D(ξ) * dxB/dY )

f_Y = divJ
      - (∂h/∂ξ)(∂ξ/∂t) * (vB/(vA+vB)-xB)
      - Dbar_Y ∇²Y
      - [ (1-h) * dxB/dY - 1 ] * ∂Y/∂t
```

Fourier/BDF1:

```text
(Y^{n+1}-Y^n)/Δt = f_Y^n - Dbar_Y k² Y^{n+1}       (S3.7)
```

Uncertainty: the docx formula extraction compresses some math. The `-[A-1] dY/dt` term is inferred from the readable S3.5/S3.6 context and from the standard algebra `A dY/dt = F -> dY/dt = F - (A-1)dY/dt`.

## 2. CUDA Actual Discrete Y Equation

The relevant implementation is:

- `compute_mu_x_kernel`: `cuda_kernels.cu:603-660`
- `compute_flux_single_component_kernel`: `cuda_kernels.cu:1241-1268`
- spectral divergence: `cuda_kernels.cu:1271-1295`
- `compute_Y_rhs_kernel`: `cuda_kernels.cu:1344-1425`
- `Y_semi_implicit_update_kernel`: `cuda_kernels.cu:1431-1450`
- Laplacian: `cuda_kernels.cu:1605-1615`
- `dYdt_prev` update: `cuda_kernels.cu:1622-1632`
- Y update driver/Picard loop: `main_cuda.cu:6957-7113`, `main_cuda.cu:7261-7266`

At each dynamics step, the code saves `phi^n` and `Y^n` at `main_cuda.cu:6257-6259`. After the phi update, the Y update uses `phi_r` as the new phi and `phi_n_saved` as old phi.

Define, at grid point `i`:

```text
x_i       = sigmoid(Y_i)
q_i       = x_i (1-x_i)
h_new_i   = h(phi_i^{n+1})
h_old_i   = h(phi_i^n)
dhdt_i    = (h_new_i - h_old_i) / Δt
term_h_i  = term_h_scale * dhdt_i * (v_B - x_i)
A_i       = (1 - h_explicit_i) * q_i
gamma_i   = A_i - 1
term_gamma_i = gamma_i * dYdt_guess_i
term_lap_i   = mean_DY * ∇²Y_i^n
```

`h_explicit_i` is `h_new_i` by default. With `--enable-Y-rhs-previous-time-level`, it becomes `h_old_i`.

The code forms:

```text
rhs_i = divJ_i - term_h_i - term_lap_i - term_gamma_i
```

The Fourier update is:

```text
Y_k^{n+1} = (Y_k^n + Δt * rhs_k) / (1 + Δt * mean_DY * k²)
```

Equivalently, in real-space operator form:

```text
(Y^{n+1}-Y^n)/Δt
  = divJ - term_h - gamma * dYdt_guess
    - mean_DY ∇²Y^n + mean_DY ∇²Y^{n+1}
```

If `dYdt_guess = (Y^{n+1}-Y^n)/Δt` is self-consistent, this becomes:

```text
A * dY/dt
  = divJ - term_h
    - mean_DY ∇²Y^n + mean_DY ∇²Y^{n+1}
```

Thus the CUDA equation matches the STO_SM S3 algebra for the `A-1` prefactor correction and the semi-implicit `Dbar_Y` stabilization.

Important discretization detail: current Picard mode only updates `dYdt_guess`. It does not recompute `xB`, `A`, `gamma`, `divJ`, `lapY`, or `mean_DY` inside the Picard loop. In Picard mode, `Y_for_Y_rhs = Y^n` explicitly (`main_cuda.cu:7041`).

## 3. Term Mapping: CUDA vs STO_SM

| CUDA term | Code | STO_SM S3 counterpart | Notes |
|---|---:|---|---|
| `xB=sigmoid(Y)` | `cuda_kernels.cu:627-630`, `1370-1382` | `Y=ln[xB/(1-xB)]` | Stable sigmoid implementation. |
| `divJ` | `main_cuda.cu:6959-7017` | `∇·M∇[...]` in S3.5/S3.6 | Code's `J` is `M∇μ`, not physical flux `-M∇μ`, so `+divJ` matches STO form. |
| `term_h` | `cuda_kernels.cu:1364-1383` | `(∂h/∂ξ)(∂ξ/∂t)(vB/(vA+vB)-xB)` | Implemented as finite-difference `dh/dt`, not old `h'(phi) dphi/dt`. |
| `gamma_local` | `cuda_kernels.cu:1385-1391` | `[(1-h) dxB/dY - 1]` | `gamma=A-1`. |
| `mean_DY` | `main_cuda.cu:7029-7034`, `cuda_kernels.cu:1845-1873` | `mean(D(ξ) dxB/dY)` | Code uses `mean(D_mix * xB(1-xB))`. |
| `term_lap` | `cuda_kernels.cu:1393-1396` | explicit `-Dbar_Y ∇²Y` in `f_Y` | Paired with implicit denominator. |
| Fourier denominator | `cuda_kernels.cu:1443-1450` | `-Dbar_Y k² Y^{n+1}` in S3.7 | Correct for `∇² -> -k²`. |
| `dYdt_prev` | `cuda_kernels.cu:1622-1632` | explicit previous-time evaluation of prefactor correction | Baseline uses previous global step derivative; Picard iterates it. |

## 4. Term_h Audit

Mixture rule in this code:

```text
c = xBtot = (1-h) xB + h v_B
```

With `v_B=1`:

```text
c = (1-h)xB + h
dc/dt = (1-h) dxB/dt + (v_B-xB) dh/dt
```

So the reaction/interface source that must be subtracted from the matrix composition equation is:

```text
term_h = (v_B - xB) * dh/dt
```

CUDA implements:

```cpp
dh_dt = (h_new - h_old) / dt;
term_h = term_h_scale * dh_dt * (v_B - xB);
rhs = divJ - term_h - term_lap - term_gamma;
```

This is at `cuda_kernels.cu:1364-1397`.

Findings:

1. No old Y-RHS form `h_prime(phi_new)*(phi_new-phi_old)/dt` remains in `compute_Y_rhs_kernel`. `h_prime_of_phi` still appears in phi RHS, volume projection, and other phi-related kernels, but not in Y RHS.

2. `term_h` uses `xB` from the Y passed to RHS. In baseline this is effectively `Y^n`, because `Y_r` has not yet been updated when the single Y solve is entered. In Picard mode it is explicitly fixed to `Y^n`. This is consistent with explicit source treatment and with the discrete local conservation numerator:

```text
Δc_from_phi at fixed xB^n = (h_new-h_old)(v_B-xB^n)
```

3. Sign sanity check, single point, no diffusion:

```text
dh/dt > 0, v_B=1, xB<1  =>  term_h > 0
rhs includes -term_h    =>  dY/dt < 0 after solving A dY/dt = -term_h
dY/dt < 0               =>  dxB/dt < 0
```

Matrix xB decreases while precipitate fraction grows, exactly the needed compensation.

4. `term_h_scale=1.5` helping drift is not evidence that `term_h` is missing a factor 1.5. With stale gamma, the code under-solves the `A dY/dt = F` relation; scaling `term_h` can compensate that lag empirically.

Conclusion: `term_h` formula is correct for the current mixture rule. I found no concrete STO_SM formula error in `term_h`.

Residual caveat: the global splitting uses `h_new-h_old` from the just-updated phi field. This is a reasonable conservative split choice, but it is not a fully coupled solve of phi and Y.

## 5. Gamma_local Audit

From `Y=logit(xB)`:

```text
dxB/dt = xB(1-xB) dY/dt
A = (1-h) xB(1-xB)
```

STO_SM's prefactor correction is:

```text
A dY/dt = F
dY/dt = F - (A-1)dY/dt
```

CUDA implements:

```cpp
logistic_deriv = xB * (1.0 - xB);
gamma_local = ((1.0 - h_for_explicit) * logistic_deriv - 1.0);
term_gamma = gamma_local * dY_dt;
rhs = ... - term_gamma;
```

This is exactly:

```text
gamma_local = A - 1
rhs = F - (A-1) dYdt_guess
```

Answers to the specific checks:

1. `gamma_local` equals `A-1`: yes.

2. `A` should not include `D_mix`, mobility, thermodynamic factor, `c_bulk`, or molar-volume correction. `A` is only the time-derivative chain-rule prefactor from `xBtot=(1-h)xB+h v_B` and `dxB/dY`. The transport coefficients belong to `divJ` and `mean_DY`, not to this prefactor.

3. The sign is consistent. Since typical `A≈0.03`, `gamma≈-0.97`. With `rhs = F - gamma*dYdt`, a self-consistent solve gives `A dYdt = F`.

4. The no-gamma result `Y_compensation_ratio≈0.027` matches `A≈xB(1-xB)≈0.029` in the matrix. Without gamma, the code effectively solves `dYdt=F`; the mass response is then only `A F` instead of `F`.

5. Picard driving the ratio toward 1 is strong evidence that the gamma formula is right and the dominant problem is stale `dYdt_guess`.

Conclusion: `gamma_local` formula is correct. I found no missing mobility/thermodynamic/c_bulk factor in gamma.

## 6. Mean_DY / Term_lap Audit

STO_SM S3.6 defines:

```text
Dbar_Y = mean( D(ξ) * exp(Y)/(1+exp(Y))² )
       = mean( D(ξ) * xB(1-xB) )
```

CUDA computes:

```cpp
Dm = D_mix(h, D_alpha, D_compound);
logistic_deriv = xB * (1.0 - xB);
mean_DY = mean(Dm * logistic_deriv);
```

This matches S3.6. It should not additionally include:

- `(1-h)`: no, the `(1-h)` time prefactor belongs to `A/gamma`, not `Dbar_Y`.
- thermodynamic factor or mobility: no, because `D(ξ)` is the chemical diffusivity used for the linear stabilizing approximation; the mobility form is already used in `divJ`.
- `c_bulk` or molar-volume correction: no, those are inside the diffusion potential `mu_x`, not the chain-rule stabilizer.

Fourier sign:

```text
lapY^n = ∇²Y^n = -k²Y^n
rhs includes -mean_DY * lapY^n
denom adds +Δt*mean_DY*k²
```

So:

```text
(Y^{n+1}-Y^n)/Δt
  = ... - mean_DY ∇²Y^n + mean_DY ∇²Y^{n+1}
```

This is the standard S3.6/S3.7 split. The `k=0` component of `∇²Y` is zero, so `term_lap` should not directly change total mass. This is consistent with diagnostics reporting mean/k0 `term_lap` near zero.

Conclusion: I do not see a clear STO_SM implementation error in `mean_DY` or `term_lap`. It is unlikely to be the main total-drift source.

Caveat: current minimal Picard does not recompute `mean_DY` or `lapY` inside iterations. That can contribute to the small residual after many Picard iterations, but it is a nonlinear/splitting residual, not an obvious S3 formula bug.

## 7. DivJ / Flux Audit

CUDA diffusion potential:

```cpp
mu_x = c_bulk * (muB - muA - c_bulk * mu_total * dVm_alpha_dxB);
mu_x += -eps_c_prime * tr(sigma)  // if elastic enabled
```

Code reference: `cuda_kernels.cu:635-658`.

Compared with S3, the constant-molar-volume limit `dVm_alpha_dxB=0` reduces to `c*(muB-muA)` plus the elastic correction. The extra `-c_bulk*mu_total*dVm_alpha_dxB` is a molar-volume correction used by this code's pseudo-binary volume model; it is not part of the compact S3 text, but it is not a Y mass-drift mechanism by itself.

Flux/divergence:

```cpp
G = gamma_thermo_nonlinear(...)
Meff = Dm / G
J_alpha = Meff * grad_mu_alpha
divJ_k += i*k_alpha*J_alpha_k
rhs += divJ
```

Code reference: `cuda_kernels.cu:1241-1295`.

This is consistent with STO_SM's `∇·M∇(diffusion potential)`. The code's `J` is the positive Onsager term `M∇μ`; it is not the physical flux `j=-M∇μ`. Therefore `rhs=+divJ` is sign-consistent with the equation.

For periodic spectral derivatives:

- At `k=0`, `i*k=0`, so `divJ_k0` is mathematically zero.
- The 2/3 dealias kernel does not remove `k=0`.
- There is no explicit k=0 overwrite that would inject a mean divergence.

Diagnostics already show `mean_divJ_effective ~ 1e-19`, so `divJ` cannot explain total `xBtot` drift. A local sign mistake would affect diffusion direction/stability, but it would still not create a net mass drift under periodic spectral divergence.

Conclusion: `divJ` is ruled out as the total drift source.

## 8. Why Picard Reduces Drift

1. `iters=1` is equivalent to baseline in the relevant sense because only one Y solve is performed using the pre-existing `dYdt_guess`. In baseline this is `dYdt_prev`; in Picard mode the initial guess is copied from `dYdt_prev`.

2. Increasing iterations monotonically reduces drift because the iteration solves the fixed-point problem:

```text
dYdt = F - (A-1)dYdt
```

with `F = divJ - term_h - mean_DY lapY + implicit-lap contribution`. As the guess approaches the current-step `dYdt`, the update approaches `A dYdt = F`, restoring the missing chain-rule amplification by roughly `1/A`.

3. No-gamma gives `ratio≈0.027` because the code then uses `dYdt≈F`, and the conserved mass response is `(1-h)xB(1-xB)dYdt≈A F`. For matrix `xB≈0.03`, `xB(1-xB)≈0.029`, matching the observed ratio.

4. At `iters=20`, `ratio≈0.985` but drift is nonzero because this is still a minimal fixed-point solve. It fixes only `gamma*dYdt`. It does not update `xB`, `A`, `divJ`, `lapY`, or `mean_DY` from the iterated Y field, and phi/Y are still operator-split.

5. The remaining drift is more likely from:

- minimal Picard with fixed `A/divJ/mean_DY/lapY`;
- phi/Y operator splitting;
- evolving nonconserved `Y` and mapping back through nonlinear sigmoid at finite `dt`.

It is less likely from:

- `term_h` formula;
- `gamma_local` formula;
- `mean_DY/term_lap` sign;
- `divJ` mean.

## 9. Is It Only dYdt_prev?

Strict answer: not proven "only". The evidence supports that stale `gamma_local*dYdt_prev` is the dominant identified source. The remaining drift after 20 Picard iterations means there are still residual discretization/splitting effects, or the minimal Picard is not solving the fully nonlinear Y equation.

I found no concrete STO_SM formula implementation error in `term_h`, `gamma_local`, `mean_DY/term_lap`, or `divJ` that would explain the observed total drift.

## 10. Source Judgment Table

| Potential source | Evidence | Status |
|---|---|---|
| scheduled insertion | `event_mass_error=0`; drift present independent of insertion events | ruled out |
| xB/Y clipping | total clip counts are zero; code diagnostics separate raw/Y-clamp/xB-clamp masses | ruled out |
| mean(divJ) / divJ sign | `mean_divJ_effective ~ 1e-19`; spectral periodic divergence has zero k0 | ruled out |
| term_h formula | Code uses `(h_new-h_old)/dt*(v_B-xB)`; no old `h_prime*dphi/dt` in Y RHS; sign sanity passes | unlikely |
| gamma_local formula | Code uses `A-1`; no-gamma ratio matches `A≈xB(1-xB)`; Picard restores ratio | unlikely |
| gamma*dYdt_prev explicit lagging | Picard iterations reduce drift monotonically and push compensation ratio to 1 | confirmed |
| mean_DY / term_lap stabilization | Formula and Fourier sign match S3; k0 contribution is zero | unlikely |
| minimal Picard fixed A/divJ/mean_DY | iters=20 still leaves finite drift while only `dYdt_guess` is updated | possible |
| phi/Y operator splitting | phi is updated before Y; Y source uses finite `h_new-h_old`; not fully coupled | possible |
| Y not being conserved variable | sigmoid update is nonlinear and not an exact conservative variable update at finite dt | possible |
| diagnostics bug | current diagnostics use same `xBtot=(1-h)xB+v_B h` and avoid double `invN`; known independent terms agree | unlikely |

## 11. Recommendations

1. Keep Picard as an optional diagnostic/accuracy control. The scan proves it targets the dominant lag.

2. Implement a full Picard option if the goal is to separate residual sources:

```text
for iter:
  recompute xB from current Y
  recompute mu_x, divJ
  recompute mean_DY and lapY if desired
  recompute A/gamma
  solve Y
```

This is the cleanest way to test whether the residual at 20 minimal iterations is from fixed coefficients.

3. For strict conservation, consider evolving/projecting the conserved field `xBtot` directly, or apply a step-end projection to match the initial total `xBtot`. Projection is an enforcement mechanism, not a formula fix.

4. Do not remove `term_gamma`. The no-gamma result is exactly what the chain-rule algebra predicts when `A≈0.03`.

5. I do not recommend changing `term_h`, `gamma_local`, or `mean_DY` based on the current evidence. There is no concrete formula bug to fix there.

