# Step 38A GP Eta Parameter Path Audit

## Scope

This audit covers the current GP `eta` Allen-Cahn path in:

- `/Users/heng/Documents/GitHub/CUDA_STO_PF/pf_params.h`
- `/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu`
- `/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu`
- `/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.h`

No solver formulas were changed in this step.

## 1. Where are `gp_W_eta`, `gp_kappa_eta`, `gp_L_eta` defined?

Parameter definitions:

- `/Users/heng/Documents/GitHub/CUDA_STO_PF/pf_params.h:22`
  - `double gp_W_eta;`
- `/Users/heng/Documents/GitHub/CUDA_STO_PF/pf_params.h:23`
  - `double gp_kappa_eta;`
- `/Users/heng/Documents/GitHub/CUDA_STO_PF/pf_params.h:24`
  - `double gp_L_eta;`

These are the only GP-eta interface / mobility parameters used by the current eta update path.

## 2. What are their defaults?

Defaults are set in `/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:4679-4681`:

```cpp
P->gp_W_eta = 0.0;
P->gp_kappa_eta = 0.0;
P->gp_L_eta = 0.0;
```

Related elastic defaults:

- `/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:4682`
  - `P->gp_eps_iso = 0.0;`
- `/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:4688`
  - `P->gp_elastic_derivative_scale = 1.0;`

## 3. Which params file / command-line path overrides them?

Param-file parser:

- `/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:7936`
  - `TRY_SET_DOUBLE("gp_W_eta", gp_W_eta);`
- `/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:7937`
  - `TRY_SET_DOUBLE("gp_kappa_eta", gp_kappa_eta);`
- `/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:7938`
  - `TRY_SET_DOUBLE("gp_L_eta", gp_L_eta);`
- `/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:7939`
  - `TRY_SET_DOUBLE("gp_eps_iso", gp_eps_iso);`
- `/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:7945`
  - `TRY_SET_DOUBLE("gp_elastic_derivative_scale", gp_elastic_derivative_scale);`

Command line:

- `/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:9040-9042`
  - `--gp-L-eta` / `--gp_L_eta`

There is no matching CLI override block for:

- `gp_W_eta`
- `gp_kappa_eta`
- `gp_eps_iso`
- `gp_elastic_derivative_scale`

So today:

- `gp_L_eta` can be overridden by param file and CLI.
- `gp_W_eta`, `gp_kappa_eta`, `gp_eps_iso`, `gp_elastic_derivative_scale` are param-file only.

## 4. Are there alternative W/kappa variables for GP eta?

No alternate GP-eta interface variables were found.

The only GP-eta nonchemical controls are:

- `gp_W_eta`
- `gp_kappa_eta`
- `gp_eps_iso`
- `gp_elastic_derivative_scale`

There are unrelated `phi` interface parameters elsewhere, but no second GP-eta `W/kappa` path.

## 5. Does `gp_W_eta` actually enter `compute_eta_rhs_kernel()`?

Yes.

Code path:

- `/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:803`
  - `double gp_eta = g_prime_of_phi(eta);`
- `/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:851`
  - `rhs_r[idx] = dgbulk_deta + gp_W_eta * gp_eta + elastic_part;`

So the double-well term is explicit and enters the real-space RHS directly.

## 6. Does `gp_kappa_eta` enter eta RHS explicitly or semi-implicitly?

Semi-implicitly.

It does not appear in the explicit real-space RHS at `/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:851`.

Instead it enters the Fourier update denominator:

- `/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:950`

```cpp
double denom = 1.0 + L_dt * kappa_eta * k2[idx];
```

and the update is:

```cpp
eta_k_new = (eta_old - L_dt * rhs) / denom;
```

So the gradient regularization is treated semi-implicitly.

## 7. Does `dgel_deta` enter eta RHS when `gp_elastic_enabled=1` and `gp_elastic_active_eta=1`?

Yes.

The eta elastic derivative is computed here:

- `/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:280-294`
  - `compute_dgel_deta_gp_point(...)`

and included in eta RHS here:

- `/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:833-838`

```cpp
elastic_part = gp_elastic_derivative_scale *
    compute_dgel_deta_gp_point(..., eps_iso_over_vB, gp_eps_iso);
```

then added into:

- `/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:851`

```cpp
rhs_r[idx] = dgbulk_deta + gp_W_eta * gp_eta + elastic_part;
```

So elastic does enter the eta equation if the elastic switches are on.

## 8. Why is `gp_eps_iso` currently 0?

Because the current defaults set:

- `/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:4682`
  - `P->gp_eps_iso = 0.0;`

and the recent Step 36 / Step 37 runs did not override it.

There is no hidden nonzero fallback for GP eigenstrain.

## 9. Confirm whether elastic is zero only because eigenstrain is zero

For the recent Step 37 audit run:

- `gp_elastic_enabled = 1`
- `gp_elastic_active_eta = 1`
- `gp_elastic_derivative_scale = 1.0`
- `gp_eps_iso = 0`

The implemented elastic derivative is:

```cpp
delta_diag = prefactor * (gp_eps_iso - eps_alpha);
return -(sigma_xx + sigma_yy + sigma_zz) * delta_diag;
```

with:

```cpp
eps_alpha = xB_alpha * eps_iso_over_vB;
```

So even with `gp_eps_iso = 0`, `dgel/deta` is not mathematically forced to zero if the hydrostatic stress and `eps_alpha` are nonzero.

However, Step 37 showed the runtime elastic interface contribution was effectively negligible:

- `eta_rhs_elastic_interface_mean = 0`
- `ratio_elastic_over_chem = 0`

Therefore the current near-zero elastic contribution is not only “because the elastic code path is disabled”:

- the code path is active
- but with `gp_eps_iso = 0` and the present stress/eigenstrain state, the elastic term is too weak to matter

## Bottom line

1. `gp_W_eta`, `gp_kappa_eta`, and `gp_L_eta` are real production parameters with a valid code path.
2. Their defaults are all zero except `gp_elastic_derivative_scale = 1`.
3. `gp_W_eta` enters eta RHS explicitly.
4. `gp_kappa_eta` enters eta update semi-implicitly.
5. `dgel/deta` enters eta RHS when GP elastic switches are enabled.
6. In the current GP-growth runs, the eta equation is chemical-drive dominated because:
   - `gp_W_eta = 0`
   - `gp_kappa_eta = 0`
   - `gp_eps_iso = 0`
   - the active elastic path remains negligible at runtime
