# 04 CUDA A–E runtime closure

`PASS_CUDA_AE_SMOKE` validates the real CUDA 96³ A–E smoke under the frozen contract. The maximum observed four-bucket residual across compact checkpoints is `1.389637712219187e-13`.

## Identity and storage controls

- Case A versus B local PF trajectory identity (all audited fields): `True`.
- Case B versus C frozen-aux local-field independence: `True`. Case C is the nonzero GP and beta-subgrid storage control.
- Case D exact matrix-to-GP transfer: `{
  "GP_inventory_delta_mol": 1.309926818847619e-21,
  "expected_exact_transfer_delta_mol": 1.309926818847619e-21,
  "matrix_inventory_delta_mol": 1.3099268188475925e-21,
  "phi_max_abs_difference": 0.0,
  "resolved_inventory_difference_mol": 0.0
}`.
- Case E closure and double-count audit: `{
  "double_count_status": "PASS_FOUR_BUCKET_SUM_AND_FIXED_RESOLVED_INVENTORY_CHECKED_AT_EVERY_CHECKPOINT",
  "frozen_auxiliary_inventory_and_PSD_persistent": true,
  "matrix_pulse": {
    "normalized_peak_minus_own_R0": 0.26354179141490564,
    "own_R0_matrix_inventory_mol": 1.296827550659144e-19,
    "peak_abs_minus_own_R0_mol": 3.417682558569151e-20,
    "peak_minus_own_R0_mol": 3.417682558569151e-20,
    "peak_step": 363,
    "peak_time_h_from_R0": 0.09991838128043773,
    "peak_time_s_from_R0": 359.70617260957584,
    "window": "ALL_OBSERVED_ACCEPTED_CHECKPOINTS"
  },
  "matrix_pulse_interpretation": "OBSERVED_RELATIVE_TO_OWN_R0; auxiliary PSD/inventory is frozen, so any recorded matrix evolution has no GP/subgrid handoff transfer route",
  "max_four_bucket_relative_residual": 1.3896377122191874e-13,
  "restart_qualifications": [
    "E 6h continuous/restart",
    "E 24h primary/6h-start checkpoint",
    "E 48h primary/6h-continuous",
    "E 48h 6h-continuous/24h-restart"
  ]
}`.

## Matrix evolution and seed behavior

Matrix pulse is always measured relative to each case's own R0 state; the Case E conditioned target is not called a pulse. The recorded audit is:

```json
{
  "A_vs_B": {
    "A_minus_B_peak_delta_mol": 0.0,
    "A_peak_step": 363,
    "B_peak_step": 363,
    "same_peak_step": true,
    "window": "ALL_OBSERVED_ACCEPTED_CHECKPOINTS"
  },
  "per_case": {
    "A": {
      "normalized_peak_minus_own_R0": 0.2507183776869433,
      "own_R0_matrix_inventory_mol": 1.309926818847619e-19,
      "peak_abs_minus_own_R0_mol": 3.284227269100935e-20,
      "peak_minus_own_R0_mol": 3.284227269100935e-20,
      "peak_step": 363,
      "peak_time_h_from_R0": 0.09991838128043773,
      "peak_time_s_from_R0": 359.70617260957584,
      "window": "ALL_OBSERVED_ACCEPTED_CHECKPOINTS"
    },
    "B": {
      "normalized_peak_minus_own_R0": 0.2507183776869433,
      "own_R0_matrix_inventory_mol": 1.309926818847619e-19,
      "peak_abs_minus_own_R0_mol": 3.284227269100935e-20,
      "peak_minus_own_R0_mol": 3.284227269100935e-20,
      "peak_step": 363,
      "peak_time_h_from_R0": 0.09991838128043773,
      "peak_time_s_from_R0": 359.70617260957584,
      "window": "ALL_OBSERVED_ACCEPTED_CHECKPOINTS"
    },
    "C": {
      "normalized_peak_minus_own_R0": 0.2507183776869433,
      "own_R0_matrix_inventory_mol": 1.309926818847619e-19,
      "peak_abs_minus_own_R0_mol": 3.284227269100935e-20,
      "peak_minus_own_R0_mol": 3.284227269100935e-20,
      "peak_step": 363,
      "peak_time_h_from_R0": 0.09991838128043773,
      "peak_time_s_from_R0": 359.70617260957584,
      "window": "ALL_OBSERVED_ACCEPTED_CHECKPOINTS"
    },
    "D": {
      "normalized_peak_minus_own_R0": 0.26354179260578875,
      "own_R0_matrix_inventory_mol": 1.296827550659143e-19,
      "peak_abs_minus_own_R0_mol": 3.4176825740128486e-20,
      "peak_minus_own_R0_mol": 3.4176825740128486e-20,
      "peak_step": 363,
      "peak_time_h_from_R0": 0.09991838128043773,
      "peak_time_s_from_R0": 359.70617260957584,
      "window": "ALL_OBSERVED_ACCEPTED_CHECKPOINTS"
    },
    "E": {
      "normalized_peak_minus_own_R0": 0.26354179141490564,
      "own_R0_matrix_inventory_mol": 1.296827550659144e-19,
      "peak_abs_minus_own_R0_mol": 3.417682558569151e-20,
      "peak_minus_own_R0_mol": 3.417682558569151e-20,
      "peak_step": 363,
      "peak_time_h_from_R0": 0.09991838128043773,
      "peak_time_s_from_R0": 359.70617260957584,
      "window": "ALL_OBSERVED_ACCEPTED_CHECKPOINTS"
    }
  },
  "seconds_per_step": 0.9909260953431841
}
```

No adapter-induced immediate seed loss is accepted; detailed component evidence is in `cuda_component_history.csv`.
