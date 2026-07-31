# Timestep diagnostic

The production candidate `dt_code=0.02` and refined `dt_code=0.01` were run to
the same code-time endpoint.

```text
dt_phi_L1=8.263201696777277e-4
dt_axis_relative=3.926911327320605e-4
dt_xB_full_field_MAE=1.114535459033156e-4
dt_xB_alpha_gt_1e-2_MAE=1.1035685456483166e-4
dt_xB_alpha_gt_1e-1_MAE=1.108364053435963e-4
dt_xB_far_matrix_MAE=1.1499033009521266e-4
mass_relative=2.701542693254547e-13
```

The main phase and particle-shape observables converge tightly, both runs are
finite, mass/zero-mode pass, and particle identities remain exact.  The
full-field xB MAE reproduces the previously diagnosed V2 value and exceeds the
historical `5e-5` threshold.

Under the frozen V1 scientific policy this xB metric is **reported but not an
initial-state identity veto**:

```text
full_field_xB_dt_MAE_blocking=false
dt_safety_status=PASS
production_dt_selection_status=REQUIRES_SEPARATE_OBSERVABLE_CONVERGENCE_AUDIT
```

This goal does not select or start a 6–48 h production timestep.

