# Cluster qualification: eta 1 nm / 4 nm unit contract

## Decision

```text
unit_contract_status=PASS_ETA_1NM_4NM_T400_CONVERTER_CONTRACT
restart_status=PASS_PF_ZERO_MODE_CUDA_CONTINUOUS_RESTART_BYTEWISE_V1
gp_paths_enabled=false
eta_dynamics_enabled=false
mass_projection_used=false
implicit_spatial_solve_used=false
```

The 1 nm grid and 4 nm interface values are the physical-unit converter
contract from the eta production branch. This test does **not** turn on GP or
eta physics; it tests the pure-PF solver under the same length/time scaling.

## Cluster evidence

- Target job: `72651`
- Tail/dependency job: `72652`
- Target state: `COMPLETED`, exit `0:0`, elapsed `00:00:58`
- Tail state: `COMPLETED`, dependency satisfied
- Output root:
  `/data/home/luozhiheng/tmp/pf_zero_mode_restart_eta1nm4nm_q11_20260728`
- Grid: 64³
- Temperature: 400 °C
- `dt_code`: 0.02
- Physical timestep: 0.8225917084910954 s
- Continuous endpoint: 96 steps = 78.96880401514517 s

The continuous run and the 48-step checkpoint/restart run produced identical
final checkpoint files:

```text
06abf1e04b689f884923a48de0e6a4d7c092c0c96509c2913f195e7662c24163  continuous_final.chk
06abf1e04b689f884923a48de0e6a4d7c092c0c96509c2913f195e7662c24163  restart_final.chk
```

The equality covers the complete serialized runtime state: `phi`, `Y`, `xB`,
`dY_dt_prev`, accepted absolute step, target mass, last lambda/residual/
derivative/iteration count, zero-mode accepted-step counter, parameter
fingerprint, backend, explicit time-level context, and reaction
discretization.

Final continuous/restarted audit:

```text
target_mass_code=7.87438977916441672e+03
final_mass_code=7.87438977916441581e+03
mean_mass_error=-3.46944695195361419e-18
last_lambda=-5.59035338992758389e-10
final_status=PASS
```

The host checkpoint test passed round-trip identity, backend mismatch,
time-level mismatch, reaction-discretization mismatch, and payload corruption
rejection.

## Source identity

```text
9bd79598fc03a5228a67a942a8d2d1867f1ecb3d7a6fa6bfb70918d5c863c3e5  main_cuda
e25de4dc23a5cbf961b0594ae40e466727bfe43c6118d24ce0119182f7457237  main_cuda.cu
f27db426148e25d6d9b221ad24e5013c522d8108c08f86db036807471faee0c6  cuda_kernels.cu
559eecf734041782c9fc9e36406358c75ad5e4ae2760d174d9cf10616840e453  pf_zero_mode_checkpoint.cpp
9d91278bd9b333a41a6d418b4ff3b46eefb2c420875cad9671d0864296e35008  pf_zero_mode_checkpoint.h
```

## Expected initialization warnings

The unchanged 0.8 nm legacy seed is smaller than this 4 nm diffuse-interface
contract, so the run reports the pre-registered under-resolution and
no-component warnings. They were not hidden, and the seed was not enlarged or
tuned. These warnings do not affect the zero-mode/restart identity result, but
this fixture is a numerical qualification case rather than a resolved
particle-growth claim.
