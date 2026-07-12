# PF Thermodynamic Derivative Tests

The host test `scripts/pf/test_project_thermodynamic_derivatives.py` reconstructs the source CALPHAD regular-solution free energy and the current production equal-volume storage:

`F(C,phi)=(1-h)f_alpha(x)+h*mu0`, `C=(1-h)x+h*v_B`.

It performs centered directional differences at fixed `phi` and fixed `C`. Expected executable identities are:

- `dF/dC = mu_B-mu_A`, matching `compute_mu_x_kernel` when all molar volumes equal one and `dVm/dx=0`.
- `dF/dphi|C = h'(mu0-mu_B)` for `v_A=0,v_B=1`, matching `compute_phi_rhs_kernel` under the same production parameters.

Command:

```bash
python3 scripts/pf/test_project_thermodynamic_derivatives.py
```

The test deliberately covers the current executable production specialization. The general unequal-volume expression is not approved by this test because the repository does not expose an independently documented authoritative conserved-density functional/restart variable for that case. That general closure remains unverified rather than being inferred from comments.
