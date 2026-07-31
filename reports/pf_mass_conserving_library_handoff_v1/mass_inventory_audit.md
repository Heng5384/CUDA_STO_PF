# Initial mass inventory audit

## Target and reconstructed ledger

```text
target_mean_C_B_tot=0.03
target_total_C_B_tot_code=26542.079999999998
actual_total_C_B_tot_code=26542.080000000005
field_relative_error=2.7412914188275474e-16
```

The target-box baseline is:

```text
derived_matrix_xB_alpha=0.006222777982325975
effective_matrix_volume_code=863565.7988914119
matrix_inventory_code=5371.878891412196
assembled_beta_inventory_code=21170.201108587797
local_delta_C_relaxation_inventory_code=-1.8993482190241315
decomposition_relative_error=1.3706457094137737e-16
```

The matrix and assembled beta ledgers sum to the target at machine precision.
No particle profile was changed to close mass.  No clipping, normalization,
full-field projection, or case-specific physical retuning was used.

The dynamic qualification retained the existing hard runtime mass gate:

```text
max_dynamic_mass_relative=2.701542693254547e-13
dynamic_mass_gate=PASS (required <=1e-10)
zero_mode_status=PASS
```

The runtime value is a drift/comparison metric over all qualification
endpoints; it does not replace the stricter initial canonical ledger above.

```text
global_inventory_status=PASS
initial_inventory_status=PASS_MACHINE_PRECISION
runtime_mass_status=PASS
```

