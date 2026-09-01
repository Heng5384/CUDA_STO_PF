# 03 CUDA A–E staged preflight

Final audit status: `PASS_CUDA_AE_SMOKE` at required stage `R4`. R0–R4 were sequenced; R4 was only launched after preceding gates passed.

- A/B/C local controls remain within `1e-14`: `True`.
- Case D zero-step transfer: matrix delta `1.309926818847593e-21` mol and GP delta `1.309926818847619e-21` mol for exact transfer `1.309926818847619e-21` mol.
- All accepted-step composition projection ledgers: `{
  "A": {
    "all_accepted_step_bound_status": "PF_CONSERVED_Y_ZERO_MODE_V1 validates Y bounds before each accepted update and rejects invalid states; no composition-clamp path is enabled",
    "all_step_Y_projection_count": 0.0,
    "all_step_phi_projection_count": 11330983429.0,
    "all_step_xB_projection_count": 0.0,
    "composition_clipping_status": "PASS_NO_COMPOSITION_PROJECTION_IN_ALL_ACCEPTED_STEPS",
    "native_coverage": "persistent native CUDA ledger across every primary accepted PF segment; R0 is zero-step",
    "phi_projection_status": "ACCOUNTED_NATIVE_PHASE_REPRESENTATION_PROJECTION_COUNTER"
  },
  "B": {
    "all_accepted_step_bound_status": "PF_CONSERVED_Y_ZERO_MODE_V1 validates Y bounds before each accepted update and rejects invalid states; no composition-clamp path is enabled",
    "all_step_Y_projection_count": 0.0,
    "all_step_phi_projection_count": 11330983429.0,
    "all_step_xB_projection_count": 0.0,
    "composition_clipping_status": "PASS_NO_COMPOSITION_PROJECTION_IN_ALL_ACCEPTED_STEPS",
    "native_coverage": "persistent native CUDA ledger across every primary accepted PF segment; R0 is zero-step",
    "phi_projection_status": "ACCOUNTED_NATIVE_PHASE_REPRESENTATION_PROJECTION_COUNTER"
  },
  "C": {
    "all_accepted_step_bound_status": "PF_CONSERVED_Y_ZERO_MODE_V1 validates Y bounds before each accepted update and rejects invalid states; no composition-clamp path is enabled",
    "all_step_Y_projection_count": 0.0,
    "all_step_phi_projection_count": 11330983429.0,
    "all_step_xB_projection_count": 0.0,
    "composition_clipping_status": "PASS_NO_COMPOSITION_PROJECTION_IN_ALL_ACCEPTED_STEPS",
    "native_coverage": "persistent native CUDA ledger across every primary accepted PF segment; R0 is zero-step",
    "phi_projection_status": "ACCOUNTED_NATIVE_PHASE_REPRESENTATION_PROJECTION_COUNTER"
  },
  "D": {
    "all_accepted_step_bound_status": "PF_CONSERVED_Y_ZERO_MODE_V1 validates Y bounds before each accepted update and rejects invalid states; no composition-clamp path is enabled",
    "all_step_Y_projection_count": 0.0,
    "all_step_phi_projection_count": 11304615220.0,
    "all_step_xB_projection_count": 0.0,
    "composition_clipping_status": "PASS_NO_COMPOSITION_PROJECTION_IN_ALL_ACCEPTED_STEPS",
    "native_coverage": "persistent native CUDA ledger across every primary accepted PF segment; R0 is zero-step",
    "phi_projection_status": "ACCOUNTED_NATIVE_PHASE_REPRESENTATION_PROJECTION_COUNTER"
  },
  "E": {
    "all_accepted_step_bound_status": "PF_CONSERVED_Y_ZERO_MODE_V1 validates Y bounds before each accepted update and rejects invalid states; no composition-clamp path is enabled",
    "all_step_Y_projection_count": 0.0,
    "all_step_phi_projection_count": 11304615121.0,
    "all_step_xB_projection_count": 0.0,
    "composition_clipping_status": "PASS_NO_COMPOSITION_PROJECTION_IN_ALL_ACCEPTED_STEPS",
    "native_coverage": "persistent native CUDA ledger across every primary accepted PF segment; R0 is zero-step",
    "phi_projection_status": "ACCOUNTED_NATIVE_PHASE_REPRESENTATION_PROJECTION_COUNTER"
  }
}`.

The phase representation projection ledger is reported separately from composition clipping; `xB` and `Y` composition projections must remain zero.
