# Workstation Parametric Nucleus Validation

Status: **generator accepted, runtime smoke blocked by CUDA driver/runtime mismatch**.

## Input

- dynamic summary: `Results/workflows/T400_xB0p030/raw/chel_T400_cuda_400x400x400_dt0.03_steps30000_r2.126nm_xB0.030/continue_dyn_1/summary.txt`
- PF param file: `Results/workflows/T400_xB0p030/raw/chel_T400_cuda_400x400x400_dt0.03_steps30000_r2.126nm_xB0.030/continue_dyn_1/pf_input.params`
- validation grid: `300 x 300 x 300`
- `dx`: `0.1 nm`

## Parametric Descriptor

- `r_eff_nm`: `5.66018`
- shape: `anisotropic`
- analytical form: `phi = exp(-(x-x0)^T A (x-x0))`
- generated amplitude: `0.753039106943`
- target phi volume: `759.590609639 nm^3`
- generated phi integral: `759.590609639 nm^3`

## Generator Acceptance

Generator result: **PASS**

- finite fields: `true`
- single connected analytical seed: `true`
- `phi_max`: `0.752861091641`
- `mass_error`: `-7.40707495339e-12`
- criterion `mass_error < 1e-10`: `true`

Generated raw bundle:

- `phi_init.raw`
- `xB_init.raw`
- `init_meta.json`
- `parametric_descriptor.json`

## Runtime Smoke

`main_cuda --init-mode raw_fields --nsteps 1` was launched on the workstation and reached raw-field initialization.

Observed initialization diagnostics:

- `phi min/max/mean = 2.72274592e-10 / 7.52861083e-01 / 2.81329855e-02`
- `xB min/max/mean = 1.70478430e-02 / 2.99999993e-02 / 1.72922319e-02`
- `hphi mean = 1.30454458e-02`
- `xBtot mean = 3.00000000e-02`
- Python metadata `mean_xBtot` difference: `-7.522e-15`

Runtime result: **BLOCKED**

Failure:

```text
CUDA driver version is insufficient for CUDA runtime version
```

This is an environment/runtime compatibility blocker, not a parametric nucleus construction failure. The smoke test did not reach the first PF timestep because the CUDA executable could not proceed after initialization.

## Acceptance Decision

Final acceptance: **false**

Reason:

```text
Acceptance policy requires generator PASS and main_cuda 1-step raw_fields smoke PASS.
Generator PASS is achieved.
Runtime PASS is blocked by CUDA driver/runtime mismatch.
```

## Next Action

Run the same `workstation_test_runner.sh` on a workstation node whose NVIDIA driver supports the CUDA runtime used by `main_cuda`, or rebuild `main_cuda` against the installed driver/runtime stack.
