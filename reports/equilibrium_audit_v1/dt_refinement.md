# Workstation dt refinement

## Authoritative branch evidence

The same clean `codex/pf-zero-mode-restart-provenance-v1` binary (`11d073a272a0b7fa668e40037bc1f96b8389909f95c74d6ad9a5db11236e01af`) was used for a T380 GP-OFF, R=8 nm nonelastic pair. The coarse run used `dt=2e-4`, 64 steps; the fine run used `dt=1e-4`, 128 steps. Both reach `t_phys=0.6341927 s` and both use `PF_CONSERVED_Y_ZERO_MODE_V1`.

At that common endpoint, both runs report `R_avg=7.779285 nm` to the printed precision. `vf_precip` is `1.050674e-03` (coarse) versus `1.054126e-03` (fine), a relative difference of 0.3275%. The zero-mode audits pass; mean mass errors are `2.28463081786145494e-15` and `5.11743425413158093e-17`, respectively. This passes the short local dt-refinement check, but it is not an equilibrium-root convergence result.

The earlier exploratory commands passed a parameter file containing `dt=0.001`, which correctly overrode their positional dt. Those runs are retained as provenance but are not called dt-refinement evidence.

## Elastic branch refinement

For R=10 nm, fixed-cell elasticity ON, the low endpoint `xAg=0.0058` was run to the same `t_phys=1.268385 s` with `dt=2e-4` (128 steps) and `dt=1e-4` (256 steps). The endpoint `vf_precip` values were `8.344152e-03` and `8.353988e-03`, respectively (relative difference 0.1177%). Both zero-mode audits passed; mean mass errors were `1.28519417330608127e-14` and `5.54223333892878144e-16`. This is a PASS for the short elastic local refinement, not a converged equilibrium-root slope.
