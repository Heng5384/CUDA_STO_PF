# IMEX-BDF2 Equation and Sign Contract

Status: `BDF2_equation_sign_status=PASS_SOURCE_AND_RUNTIME_QUALIFIED`

This contract is recovered from the accepted Lie-BE implementation and the
underlying CUDA kernels. BDF2 is permitted to replace only the two time
derivatives. It must not change any thermodynamic, mobility, flux, divergence,
phase-force, elasticity, or gradient-energy sign.

## Conserved Composition

The positive-face flux stored by the FV kernel is

```text
J_{i+1/2} = M_{i+1/2} (mu_{i+1} - mu_i) / dx,
M_{i+1/2} = harmonic_mean(M_i,M_{i+1}) >= 0.
```

Source: `cuda_kernels.cu:3699-3750`. In particular, the source assigns the
positive sign at line 3743; this project calls `M grad(mu)` the flux.

The cell divergence is the positive-face incidence

```text
(div J)_i = (J_{i+1/2}-J_{i-1/2})/dx
          + (J_{j+1/2}-J_{j-1/2})/dy
          + (J_{k+1/2}-J_{k-1/2})/dz.
```

Source: `cuda_kernels.cu:3904-3924`. Periodicity makes the global sum of this
operator zero to reduction roundoff.

The accepted Lie-BE residual is

```text
R_C^BE = C_np1 - C_n - dt*divJ(C_np1,phi_n).
```

Source: `cuda_kernels.cu:3494-3501`. Therefore the source convention is

```text
dC/dt = div(M grad(mu)).
```

The BDF2 residual must consequently be

```text
R_C^BDF2 = (3 C_np1 - 4 C_n + C_nm1)/(2 dt)
            - divJ(C_np1,phi_E) = 0,
phi_E = 2 phi_n - phi_nm1.
```

The algebraically equivalent residual used by the nonlinear solver is

```text
C_anchor = (4 C_n-C_nm1)/3,
dt_BDF2 = 2 dt/3,
R_C_scaled = C_np1-C_anchor-dt_BDF2*divJ(C_np1,phi_E).
```

Multiplication by `3/(2dt)>0` recovers the rate-form residual, so roots and
signs are identical.

## Phase Field

`compute_phi_rhs_kernel` constructs the local free-energy derivative

```text
g_local = W*g'(phi) + chemical derivative + elastic derivative.
```

Source: `cuda_kernels.cu:399-560`. The spectral semi-implicit Lie-BE update is

```text
phi_np1 = [phi_n-dt*Lphi*g_local] /
           [1+dt*Lphi*kappa_phi*k^2].
```

Source: `cuda_kernels.cu:763-785`. Because Fourier `lap(phi)=-k^2 phi`, this
is equivalent to

```text
(phi_np1-phi_n)/dt
+ Lphi*[g_local(phi_np1,C_fixed)-kappa_phi*lap(phi_np1)] = 0.
```

The fixed-C PDAS cold residual uses exactly this sign:

```text
energy_gradient = g_explicit-kappa_phi*lap_phi
R_phi^BE = (phi_np1-phi_n)/dt + Lphi*energy_gradient.
```

Source: `main_cuda.cu:3116-3153`.

The BDF2 phase residual is therefore frozen as

```text
R_phi^BDF2 = (3 phi_np1-4 phi_n+phi_nm1)/(2 dt)
             + Lphi*[g_local(phi_np1,C_np1)
                     -kappa_phi*lap(phi_np1)] = 0.
```

The free-set Jacobian time diagonal is `3/(2dt)`, implemented by passing
`phase_rate_dt=2dt/3` to the unchanged PDAS Jacobian/preconditioner kernels.

## Frozen Rules

1. `J=M grad(mu)` remains unchanged.
2. `dC/dt=div(J)` remains unchanged.
3. `delta F/delta phi=g_local-kappa*lap(phi)` remains unchanged.
4. `dphi/dt=-Lphi*delta F/delta phi` remains unchanged.
5. BDF2 changes only the two time-rate formulas and supplies the explicit
   second-order phase context to transport.
6. `C_np1` is fixed throughout the phase PDAS solve.
7. No final transport correction, M3, Anderson, second phase solve, clipping,
   or physical mass projection belongs to this contract.



## Runtime Closure

The consistent-history ladder recovered second-order ratios in `[3,5]` for
the registered `Ctot`, `phi`, and `h` observables. All accepted fixed-step
runs retained the source signs above; no residual or physical parameter was
changed during qualification.
