Step 17 conversion mass audit report

Scope

- This round did not modify the Step 16B stochastic trigger logic.
- It did not modify `storage_exact`, the eta PDE, the transport solver, or the GP->beta conversion operator itself.
- It only added:
  - conversion-stage mass audit diagnostics
  - stop-after-first-conversion audit mode
  - audit CSV output
  - a helper script to rerun the three audit cases

Modified files

- [/Users/heng/Documents/GitHub/CUDA_STO_PF/pf_params.h](/Users/heng/Documents/GitHub/CUDA_STO_PF/pf_params.h)
- [/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu)
- [/Users/heng/Documents/GitHub/CUDA_STO_PF/run_step17_conversion_mass_audit.sh](/Users/heng/Documents/GitHub/CUDA_STO_PF/run_step17_conversion_mass_audit.sh)

New parameters

- `gp_to_beta_conversion_mass_audit_enabled`
- `gp_to_beta_stop_after_conversion_audit`
- `gp_to_beta_conversion_audit_prefix`

New CSV

- `gp_to_beta_conversion_mass_audit.csv`

Case paths

- Case A, forced `P=1`, stop-after-conversion:
  - `/tmp/step17_conversion_audit/runs/p1_stop/Results/chel_T400_cuda_32x32x32_dt1e-05_steps50_xB0.030/P1_stop_after_conversion`
- Case B, forced `P=1`, normal 50-step run with audit:
  - `/tmp/step17_conversion_audit/runs/p1_50/Results/chel_T400_cuda_32x32x32_dt1e-05_steps50_xB0.030/P1_50steps_audit`
- Case C, deterministic Step 15 forced-event with audit:
  - `/tmp/step17_conversion_audit/runs/det_50/Results/chel_T400_cuda_32x32x32_dt1e-05_steps50_xB0.030/deterministic_50steps_audit`

Case status

- Case A: success
- Case B: success
- Case C: success

Default-path regression

- With audit disabled, the updated code reproduces the old Step 16B `P=1` outputs exactly.
- `phi_50.vtk`, `xB_50.vtk`, `xBtot_gp_50.vtk`, `eta_50.vtk` MD5 all match the pre-audit Step 16B run.

Deterministic vs stochastic `P=1`

- Case B and Case C are field-identical.
- `phi_50.vtk`, `xB_50.vtk`, `xBtot_gp_50.vtk`, `eta_50.vtk` MD5 all match between stochastic `P=1` and deterministic forced-event.
- Therefore the Step 16B trigger layer is not the source of the later drift.

Stage mass table

Case A/B/C have the same accepted conversion audit stages at `step=1`.

| stage | delta_sum_xBtot_from_before_conversion | delta_patch_mass | clipped_mass_loss | num_clipped_low | num_clipped_high |
|---|---:|---:|---:|---:|---:|
| `before_conversion` | `0` | `0` | `0` | `0` | `0` |
| `after_phi_eta_update` | `+1.74269943402919125e+01` | `+1.74268847516975238e-02` | `0` | `0` | `0` |
| `after_xB_or_xBtot_update` | `+1.74269943402919125e+01` | `+1.74268847516975238e-02` | `0` | `0` | `0` |
| `after_compensation` | `+2.52181123414629837e-05` | `-1.30104260698260532e-17` | `0` | `0` | `0` |
| `after_Y_reconstruction` | `+2.52181123414629837e-05` | `-1.30104260698260532e-17` | `0` | `0` | `0` |
| `after_clipping` | `+2.52181123414629837e-05` | `-1.30104260698260532e-17` | `0` | `200` | `0` |
| `end_of_event_step` | `+2.52181123414629837e-05` | `-1.30104260698260532e-17` | `0` | `200` | `0` |

Interpretation

- The raw conversion geometry change appears at `after_phi_eta_update`, as expected.
- The local shell compensation then removes essentially all of that patch-scale mass change.
- Patch mass after compensation is conserved to numerical precision:
  - `mass_error_comp = -1.3010426070e-17`
- The remaining global offset immediately after conversion is only:
  - `delta_sum_xBtot_from_before_conversion = 2.52181123414629837e-05`
  - relative to total mass, this is about `2.56e-08`
- That is far below the later 50-step drift level of `6.115772602651491e-03`.

Stop-after-conversion result

- Case A stops immediately after the first accepted event and writes:
  - `phi_after_conversion.vtk`
  - `eta_after_conversion.vtk`
  - `xB_after_conversion.vtk`
  - `xBtot_gp_after_conversion.vtk`
- Case A summary:
  - `total_relative_drift = 2.34346060327606581e-08`
  - no clipping in the post-event PDE chain because the PDE chain never runs after the event

This shows the conversion instant itself is nearly conservative.

Clipping

- Event-internal clipping is present in the shell compensation path:
  - `num_clipped_low = 200`
  - `num_clipped_high = 0`
- But the event-internal conserved-mass difference due to that clipping is:
  - `clipped_mass_loss = 0`
- So the event-local clipping inside conversion is not the main source of the later `6.1e-3` drift.

Storage residual

- `storage_residual_mean = 0`
- `storage_residual_max_abs = 0`
- `storage_residual_sum = 0`

Reason:

- During dynamics there is no independently stored `xBtot` field that diverges from the reconstructed `storage_exact` relation.
- The audit therefore reconstructs `xBtot_gp` using the same host-side formula as the main code, so the residual is identically zero by construction.
- This means the current audit can rule out an internal mismatch in the host-side conversion state, but it does not detect a separate hidden `xBtot` state because none exists here.

50-step reproduction

- Case B reproduces the known forced-event drift:
  - `total_relative_drift = 6.11577260265149143e-03`
  - `suspected_primary_source = Y_update_k0_drift`
  - `total_clip_count_xB = 34`
- Case C reproduces the same numbers to numerical precision.

Where the drift first appears

- Not at `after_compensation`
- Not at `after_clipping`
- Not at `end_of_event_step`

The first large drift does not appear inside the conversion audit stages.
It appears later in the post-conversion PDE evolution, and the existing summary still identifies the dominant accumulated source as:

- `Y_update_k0_drift`

Conclusion

- Problem source:
  - not Step 16B stochastic trigger
  - not the conversion instant itself at leading order
  - not event-local conserved-mass loss from conversion clipping
  - most likely post-conversion dynamics, with the dominant accumulated source in the subsequent `Y` update chain

More precise statement

- Case A shows the conversion operator itself is nearly mass-conserving.
- Case B/C show the full 50-step drift reappears only when post-conversion PDE evolution proceeds.
- Therefore the current evidence points to:
  - `problem source = post-conversion dynamics`
  - with `Y_update_k0_drift` as the primary accumulated channel
  - and not a first-order mass defect introduced at the conversion instant

Next suggestion

- Keep Step 16B frozen.
- Move the next diagnostic round to the first few PDE steps after conversion:
  - compare `before_Y`, `after_Y`, and clipping terms starting from the accepted conversion state
  - isolate whether the first nontrivial post-event mass jump is driven by `divJ`, `Y_update`, or later field clipping
