# Step 34 GP Interface Driving Balance Report

- `result_dir = /tmp/step34_gp_interface_balance/runs/step34_gp_interface_balance_xB05_eta1_R1_dep2_N96/Results/chel_T380_cuda_96x96x96_dt0.01_steps5000_xB0.030/step34_gp_interface_balance_xB05_eta1_R1_dep2_N96`
- `step = 5000`
- `interface eta window = [0.1, 0.9]`
- `interface_voxel_count = 5506`

## GP Eta Parameters

- `gp_W_eta = 0.0000000000e+00`
- `gp_kappa_eta = 0.0000000000e+00`
- `gp_L_eta = 0.0000000000e+00`

## Interface Means

- `eta_rhs_chem_interface_mean = -2.8302932097e+02`
- `eta_rhs_dw_interface_mean = 0.0000000000e+00`
- `eta_rhs_elastic_interface_mean = -3.7086814384e-06`
- `eta_rhs_grad_interface_mean = 0.0000000000e+00`
- `eta_rhs_net_explicit_interface_mean = -2.8302932495e+02`
- `eta_rhs_full_variational_interface_mean = -2.8302932495e+02`
- `eta_evolution_drive_interface_mean = 2.8302932495e+02`

## Interpretation

- Dominant interface-balance term by magnitude: `chemical`.
- `eta_rhs_full_variational` is the local total free-energy derivative entering gradient flow.
- The actual local eta evolution drive is `-eta_rhs_full_variational`.
- `gp_L_eta = 0`, so even a nonzero local eta driving force does not produce meaningful deterministic eta evolution in this run.
- `gp_W_eta = 0` and `gp_kappa_eta = 0`, so this test has no eta double-well or gradient regularization; the interface-balance reduces almost entirely to the chemical term.
- The interface net drive is not near zero; check radial profile for where cancellation fails.
