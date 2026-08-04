# 400-cube fixed-xB=0.03 elastic PSD-sign pilot — V5 static-preparation registry

**Status:** `RUNNING_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_PILOT_V1`  
**Preparation job:** `73849` — `COMPLETED`  
**Production job:** `73850` — `RUNNING` (`gpu_uvip`, 1 GPU, 4 CPUs, `mem=0`, 48 h)  
**Read-only tail job:** `73851` — `PENDING`, dependency `afterany:73850`.

## Preserved preceding evidence

- V3 preparation job `73798` built all seven native `160^3` elastic profiles and then failed before fixture creation because a legacy verifier imposed an unrelated `96^3` native-shape check.
- V4 preparation job `73847` used the repaired shape contract and failed before fixture creation because pure-beta core voxels have the physically correct `alpha=1-h=0`; the old builder incorrectly demanded nonzero matrix fraction in those voxels.
- Neither V3 nor V4 created a production allocation, modified a historical result, or altered physical parameters. Their non-overwriting roots remain preserved.

The reused V3 profile library is read-only and hash-pinned:

```text
library_root=/data/home/luozhiheng/tmp/pf_400cube_fixed_xb03_elastic_psd_sign_prepare_v3_20260802/profile_library/library
library_manifest_sha256=a1b0dba740113cdfa97cd86c762030fe190fe23036487a82afff6df7770118da
native_grid=160x160x160
registered_radii_nm=17.0,17.5,18.0,18.5,19.0,19.5,20.0
all_profile_final_audits=PASS
```

## V5 numerical-only repair contract

1. The generic verifier now accepts a native shape supplied explicitly by the 400-cube fixture specification. Old callers retain the fixed `96^3` default.
2. The quintic (h(\phi)) interpolation is evaluated with an algebraically identical endpoint-stable form, so `h(0)=0` and `h(1)=1` exactly in binary64.
3. The portable matrix field `delta_C_relaxation/(1-h)` is evaluated only where `1-h>1e-12`. A hard gate rejects any pure-beta-core `delta_C_relaxation` above `1e-14`; the measured maximum is `6.279385872517745e-17`.

These changes do not change the profile fields, total inventory, PSD, PF physics, elasticity, time mapping, or acceptance thresholds. They introduce no clipping, resampling, interpolation, analytic particle, or post-assembly mass projection.

## V5 submission contract

```text
source=/data/home/luozhiheng/tmp/codex_pf_400cube_fixed_xb03_elastic_psd_sign_v5_20260802
preparation_root=/data/home/luozhiheng/tmp/pf_400cube_fixed_xb03_elastic_psd_sign_prepare_v5_20260802
reserved_nonoverwriting_production_root=/data/home/luozhiheng/tmp/pf_400cube_6h48h_fixed_xb03_elastic_psd_sign_pilot_v5_20260802
preparation_job_id=73849
fixture_spec=fixture_spec_native160_v1.json
fixture_spec_sha256=a87daffc3816bebcff7c8735aace3e00ab40100649383fe4f7a3c2c958994410
profile-library reuse=read-only, hash-pinned V3 library
preparation_sbatch_sha256=03eaba8a9ee467237a3c2dc7bb29ad85e08c4723146951bc516e92bd3d4dbd16
```

## Static gates and production registration

All prerequisite markers are exact PASS:

```text
fixture_status=PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_FIXTURE_V1
static_audit_status=PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_FIXTURE_STATIC_V1
production_preflight_status=PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_PRODUCTION_PREFLIGHT_V1
```

The completed fixture has 64 particles and reports:

```text
mean_C_Btot=0.030000000000000006
beta_h_volume_nm3=1531493.6800361243
far_field_xAg_h_lt_1e-4=0.0062012154792882475
fixture_manifest_sha256=b8b59357b659078e51077330f4fe1c31caf302c2605e809c71d8e41b18771b13
```

The discrete beta-inventory relative deviation is about `1.84e-6`, below the
frozen `1e-4` acceptance limit; the far-field Ag value lies in the frozen
`[0.0058, 0.0066]` interval. The production registration is:

```text
production_job_id=73850
tail_job_id=73851
output_root=/data/home/luozhiheng/tmp/pf_400cube_6h48h_fixed_xb03_elastic_psd_sign_pilot_v5_20260802
sbatch_sha256=ea490d4f6bcb17ab9970dd84ea3dba0a71981a3e4dbb499a861b106e08c146b0
binary_sha256=759956a89780db9a19ccd51b4115319de47463a3fd7b8d446a022a74c9beeb3b
parameter_sha256=3571c917cb5696f85ff5a2dbe9d43e7c7ed6c1634c1b13f9a2777a0b2aebadc3
fixture_sha256=b8b59357b659078e51077330f4fe1c31caf302c2605e809c71d8e41b18771b13
profile_library_sha256=a1b0dba740113cdfa97cd86c762030fe190fe23036487a82afff6df7770118da
checkpoint_schedule_sha256=fc593341b62d806d069066261baf8d18cbb58ce97538b6c8cad1ba29d2e7081d
```

The production contract remains one unique `gpu_uvip/gpu_uvip` 48 h allocation,
`dt_code=0.02`, `dt_physical_s=0.9909260953431841`, `final_step=152585`, 400³,
elasticity enabled, GP/new-beta/external-source paths disabled, one-hour checkpoints,
and no bulk VTK output.

## First registered one-hour checkpoint

The first completed production segment is independently qualified as:

```text
segment=step_3633
segment_status=PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_SEGMENT_V1
checkpoint=step_3633.chk
checkpoint_sha256=c32e297aafd5c79a52f4ee2831a47c77e6aef4887546985be0537438628374ff
zero_mode_status=PASS
target_mass_code=1.92000000000000000e+06
final_mass_code=1.92000000000000000e+06
mean_mass_error=0
last_zero_mode_lambda=5.77235563661443263e-12
elastic_solver_status=PASS_ELASTIC_WARM_START_RESIDUAL_V1
elastic_nonconverged_steps=0
elastic_mean_iterations=2.552160749
segment_wall_time_s=1274.271
```

Both the segment stderr and driver stderr were empty. The measured initial
throughput is about `0.3507 s/step`; this is operational evidence only, not a
claim that the throughput or microstructure trend will remain unchanged through
48 h. The driver has continued from this checkpoint to the next registered
endpoint (`7266`).

## Second registered one-hour checkpoint

The `step_7266` segment independently reports:

```text
segment_status=PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_SEGMENT_V1
checkpoint=step_7266.chk
checkpoint_sha256=18069e50b75d071d91dda4462c83d98f56f5973f67a955e3fd22296d57a34b6c
zero_mode_status=PASS
target_mass_code=1.92000000000000000e+06
final_mass_code=1.92000000000001490e+06
mean_mass_error=2.32830643653869618e-16
last_zero_mode_lambda=0
elastic_solver_status=PASS_ELASTIC_WARM_START_RESIDUAL_V1
elastic_nonconverged_steps=0
elastic_mean_iterations=2.000000000
```

Its segment stderr and the driver stderr remain empty. This is only the second
hourly numerical-integrity checkpoint; the single production job remains
running, and no PSD, lineage, or thermal-trend conclusion is yet registered.

## Third registered one-hour checkpoint

The `step_10899` segment independently reports:

```text
segment_status=PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_SEGMENT_V1
checkpoint=step_10899.chk
checkpoint_sha256=987d5d84f7a30be4ef5a4ee8e5ed3c470665ea83b705a2e43da899ac33721ce2
zero_mode_status=PASS
target_mass_code=1.92000000000000000e+06
final_mass_code=1.91999999999999953e+06
mean_mass_error=-7.27595761418342557e-18
last_zero_mode_lambda=9.99323722599212514e-12
elastic_solver_status=PASS_ELASTIC_WARM_START_RESIDUAL_V1
elastic_nonconverged_steps=0
elastic_mean_iterations=2.785026149
elastic_last_residual=3.81400348790993147e-07
```

The segment and driver stderr logs remain empty. The driver has continued from
this checkpoint to `step_14532`; these facts establish early numerical
integrity only and do not constitute a 48 h microstructure or transport result.

## Fourth registered one-hour checkpoint

The `step_14532` segment independently reports:

```text
segment_status=PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_SEGMENT_V1
checkpoint=step_14532.chk
checkpoint_sha256=e3a08ae6545d82a3bef55dbd6c99d8b5712b75f741e753be6bcda03494f07229
zero_mode_status=PASS
target_mass_code=1.92000000000000000e+06
final_mass_code=1.92000000000000000e+06
mean_mass_error=0
last_zero_mode_lambda=2.29779068994222925e-10
elastic_solver_status=PASS_ELASTIC_WARM_START_RESIDUAL_V1
elastic_nonconverged_steps=0
elastic_mean_iterations=3.000000000
elastic_last_residual=3.28219959891018171e-07
```

The segment and driver stderr logs are empty. The production driver has resumed
from this checkpoint at `step_18165`; no 48 h structural or transport conclusion
is registered before the complete chain and final postprocessing pass.

## Fifth registered one-hour checkpoint

The `step_18165` segment completed and the driver has restarted from it toward
`step_21798`. Its independent numerical evidence is:

```text
segment_status=PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_SEGMENT_V1
checkpoint=step_18165.chk
checkpoint_sha256=94eacf290d9a97f165f762cbc00af4d4fb4ba47e12552dc4c719f0a576d9da9b
zero_mode_status=PASS_PF_CONSERVED_Y_ZERO_MODE_V1
target_mass_code=1.92000000000000000e+06
final_mass_code=1.91999999999999977e+06
mean_mass_error=-3.63797880709171279e-18
last_zero_mode_lambda=1.38992016306014153e-10
checkpoint_restart_provenance=RESTORED_AND_VALIDATED
elastic_solver_status=PASS_ELASTIC_WARM_START_RESIDUAL_V1
elastic_nonconverged_steps=0
elastic_mean_iterations=3.000000000
elastic_last_residual=3.27212251618144464e-07
segment_wall_seconds=1324.405
segment_mean_wall_seconds_per_step=0.360767
reported_peak_memory_GB=17.69
```

The segment stderr and the new `step_21798` segment stderr are both empty.
The reported VTK paths are explicit suppression notices; no bulk VTK field was
written. This remains early numerical-integrity evidence only, not a completed
48 h structural or transport conclusion.

## Sixth registered one-hour checkpoint

The `step_21798` segment (approximately 12 h) independently reports:

```text
segment_status=PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_SEGMENT_V1
checkpoint=step_21798.chk
checkpoint_sha256=b9cf2c34441353ca7d6d1697f3709d619ce3bb28d3c53625b34e9cc4740e0469
zero_mode_status=PASS_PF_CONSERVED_Y_ZERO_MODE_V1
target_mass_code=1.92000000000000000e+06
final_mass_code=1.91999999999999977e+06
mean_mass_error=-3.63797880709171279e-18
last_zero_mode_lambda=1.75008487189232911e-11
checkpoint_restart_provenance=RESTORED_AND_VALIDATED
elastic_solver_status=PASS_ELASTIC_WARM_START_RESIDUAL_V1
elastic_nonconverged_steps=0
elastic_mean_iterations=3
elastic_last_residual=3.54327388899547215e-07
segment_wall_seconds=1325.365
```

The segment and driver stderr logs remain empty. The production driver has
continued into the `step_25431` segment. This is early numerical-integrity
evidence only; it does not establish a 48 h microstructure or transport result.

## Seventh registered one-hour checkpoint

The `step_25431` segment (approximately 13 h) independently reports:

```text
segment_status=PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_SEGMENT_V1
checkpoint=step_25431.chk
checkpoint_sha256=ea7ebabb71804fca039a066c527e2e2d19838b7c502ef09b9ee2c919f807f54c
zero_mode_status=PASS_PF_CONSERVED_Y_ZERO_MODE_V1
target_mass_code=1.92000000000000000e+06
final_mass_code=1.92000000000000000e+06
mean_mass_error=0.00000000000000000e+00
last_zero_mode_lambda=-5.85924929899351371e-11
checkpoint_restart_provenance=RESTORED_AND_VALIDATED
elastic_solver_status=PASS_ELASTIC_WARM_START_RESIDUAL_V1
elastic_nonconverged_steps=0
elastic_mean_iterations=3.000000000
elastic_last_residual=3.91408931778244418e-07
segment_wall_seconds=1325.970
segment_mean_wall_seconds_per_step=0.361207
reported_peak_memory_GB=17.69
```

The segment and driver stderr logs remain empty. Its reported VTK paths are
suppression notices, not bulk-field output. The driver has resumed from this
checkpoint; this remains numerical-integrity evidence only and does not
establish a 48 h microstructure or transport result.

## Eighth registered one-hour checkpoint

The `step_29064` segment (approximately 14 h) independently reports:

```text
segment_status=PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_SEGMENT_V1
checkpoint=step_29064.chk
checkpoint_sha256=91e1029a2130986d6a8624730052aae442887efce345eb9c42ac1dabbb52f3e6
zero_mode_status=PASS_PF_CONSERVED_Y_ZERO_MODE_V1
target_mass_code=1.92000000000000000e+06
final_mass_code=1.92000000000000000e+06
mean_mass_error=0.00000000000000000e+00
last_zero_mode_lambda=-2.72652403729504141e-11
checkpoint_restart_provenance=RESTORED_AND_VALIDATED
elastic_solver_status=PASS_ELASTIC_WARM_START_RESIDUAL_V1
elastic_nonconverged_steps=0
elastic_mean_iterations=3
elastic_last_residual=4.55288361710837811e-07
segment_wall_seconds=1309.685
segment_mean_wall_seconds_per_step=0.360497
reported_peak_memory_GB=17.69
```

The segment and driver stderr logs remain empty. Its reported VTK paths are
suppression notices, not bulk-field output. The driver has resumed from this
checkpoint; this remains numerical-integrity evidence only and does not
establish a 48 h microstructure or transport result.

## Ninth registered one-hour checkpoint

The `step_32697` segment (approximately 15 h) independently reports:

```text
segment_status=PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_SEGMENT_V1
checkpoint=step_32697.chk
checkpoint_sha256=e9157117506a04b7880ecdcbfb00a4c137f33ad8fcf29c8ed4f42b02a20eea39
zero_mode_status=PASS_PF_CONSERVED_Y_ZERO_MODE_V1
target_mass_code=1.92000000000000000e+06
final_mass_code=1.92000000000000000e+06
mean_mass_error=0.00000000000000000e+00
last_zero_mode_lambda=2.28253918187494044e-09
checkpoint_restart_provenance=RESTORED_AND_VALIDATED
elastic_solver_status=PASS_ELASTIC_WARM_START_RESIDUAL_V1
elastic_nonconverged_steps=0
elastic_mean_iterations=3
elastic_last_residual=6.87372362654252257e-07
segment_wall_seconds=1310.813
segment_mean_wall_seconds_per_step=0.360807
reported_peak_memory_GB=17.69
```

The segment and driver stderr logs remain empty. Its reported VTK paths are
suppression notices, not bulk-field output. The driver has resumed from this
checkpoint; this remains numerical-integrity evidence only and does not
establish a 48 h microstructure or transport result.

## Tenth registered one-hour checkpoint

The `step_36330` segment (approximately 16 h) independently reports:

```text
segment_status=PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_SEGMENT_V1
checkpoint=step_36330.chk
checkpoint_sha256=911cd636b32f1c41ce84d73a90c63640b784a4bc8778b9d3d353f9eaec01ce92
zero_mode_status=PASS_PF_CONSERVED_Y_ZERO_MODE_V1
target_mass_code=1.92000000000000000e+06
final_mass_code=1.92000000000000000e+06
mean_mass_error=0.00000000000000000e+00
last_zero_mode_lambda=1.58670315016957542e-09
checkpoint_restart_provenance=RESTORED_AND_VALIDATED
elastic_solver_status=PASS_ELASTIC_WARM_START_RESIDUAL_V1
elastic_nonconverged_steps=0
elastic_mean_iterations=3
elastic_last_residual=4.26463449116233394e-07
segment_wall_seconds=1311.588
segment_mean_wall_seconds_per_step=0.361021
reported_peak_memory_GB=17.69
```

The segment and driver stderr logs remain empty. Its reported VTK paths are
suppression notices, not bulk-field output. The driver has resumed from this
checkpoint; this remains numerical-integrity evidence only and does not
establish a 48 h microstructure or transport result.

## Eleventh registered one-hour checkpoint

The `step_39963` segment (approximately 17 h) independently reports:

```text
segment_status=PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_SEGMENT_V1
checkpoint=step_39963.chk
checkpoint_sha256=814138364b8baf3d26adf57fc0ec9b7927aaa20a0ff15577fee4e485937e4272
zero_mode_status=PASS_PF_CONSERVED_Y_ZERO_MODE_V1
target_mass_code=1.92000000000000000e+06
final_mass_code=1.92000000000000000e+06
mean_mass_error=0.00000000000000000e+00
last_zero_mode_lambda=-9.75207175307529340e-12
checkpoint_restart_provenance=RESTORED_AND_VALIDATED
elastic_solver_status=PASS_ELASTIC_WARM_START_RESIDUAL_V1
elastic_nonconverged_steps=0
elastic_mean_iterations=3
elastic_last_residual=4.79256439537517968e-07
segment_wall_seconds=1324.290
segment_mean_wall_seconds_per_step=0.364517
```

The segment and driver stderr logs remain empty. Its reported VTK paths are
suppression notices, not bulk-field output. The driver has resumed from this
checkpoint; this remains numerical-integrity evidence only and does not
establish a 48 h microstructure or transport result.

## Twelfth registered one-hour / 18 h science checkpoint

The `step_43596` segment (the frozen 18 h science endpoint) independently
reports:

```text
segment_status=PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_SEGMENT_V1
checkpoint=step_43596.chk
checkpoint_sha256=27d2801f67f5809ee09320e1b8e6c29da941d5da977efc6ddddaf28dc53cb757
zero_mode_status=PASS_PF_CONSERVED_Y_ZERO_MODE_V1
target_mass_code=1.92000000000000000e+06
final_mass_code=1.91999999999999953e+06
mean_mass_error=-7.27595761418342557e-18
last_zero_mode_lambda=9.20093789439046313e-12
checkpoint_restart_provenance=RESTORED_AND_VALIDATED
elastic_solver_status=PASS_ELASTIC_WARM_START_RESIDUAL_V1
elastic_nonconverged_steps=0
elastic_mean_iterations=3
elastic_last_residual=5.47462523155116350e-07
segment_wall_seconds=1325.068
segment_mean_wall_seconds_per_step=0.364731
```

The segment and driver stderr logs remain empty. Its reported VTK paths are
suppression notices, not bulk-field output. The driver has resumed from this
checkpoint; this remains numerical-integrity evidence only and does not
establish a 48 h microstructure or transport result.

## Thirteenth registered one-hour checkpoint

The `step_47229` segment (approximately 19 h) independently reports:

```text
segment_status=PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_SEGMENT_V1
checkpoint=step_47229.chk
checkpoint_sha256=b9ae63d75a77991079432f248e9657e9a079a8d04e232d71a56f7854962648f7
zero_mode_status=PASS_PF_CONSERVED_Y_ZERO_MODE_V1
target_mass_code=1.92000000000000000e+06
final_mass_code=1.92000000000000000e+06
mean_mass_error=0.00000000000000000e+00
last_zero_mode_lambda=1.28078347919849651e-10
checkpoint_restart_provenance=RESTORED_AND_VALIDATED
elastic_solver_status=PASS_ELASTIC_WARM_START_RESIDUAL_V1
elastic_nonconverged_steps=0
elastic_mean_iterations=3
elastic_last_residual=5.74291806681941649e-07
segment_wall_seconds=1325.555
segment_mean_wall_seconds_per_step=0.364865
```

The segment and driver stderr logs remain empty. Its reported VTK paths are
suppression notices, not bulk-field output. The driver has resumed from this
checkpoint; this remains numerical-integrity evidence only and does not
establish a 48 h microstructure or transport result.

## Fourteenth registered one-hour checkpoint

The `step_50862` segment (approximately 20 h) independently reports:

```text
segment_status=PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_SEGMENT_V1
checkpoint=step_50862.chk
checkpoint_sha256=a28af8658b5b0c3207de96c542355c71d8177b21bc837ff3b02b52a38609d8e3
zero_mode_status=PASS_PF_CONSERVED_Y_ZERO_MODE_V1
target_mass_code=1.92000000000000000e+06
final_mass_code=1.92000000000000000e+06
mean_mass_error=0.00000000000000000e+00
last_zero_mode_lambda=1.47022243821484707e-09
checkpoint_restart_provenance=RESTORED_AND_VALIDATED
elastic_solver_status=PASS_ELASTIC_WARM_START_RESIDUAL_V1
elastic_nonconverged_steps=0
elastic_mean_iterations=3
elastic_last_residual=6.27125232078676278e-07
segment_wall_seconds=1323.924
segment_mean_wall_seconds_per_step=0.364416
```

The segment and driver stderr logs remain empty. Its reported VTK paths are
suppression notices, not bulk-field output. The driver has resumed from this
checkpoint; this remains numerical-integrity evidence only and does not
establish a 48 h microstructure or transport result.

## Fifteenth registered one-hour checkpoint

The `step_54495` segment (approximately 21 h) independently reports:

```text
segment_status=PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_SEGMENT_V1
checkpoint=step_54495.chk
checkpoint_sha256=08480df60aa4351b5f2c0dba0f4d58c73ad0dfa1d1f197fb44482232592f44db
zero_mode_status=PASS_PF_CONSERVED_Y_ZERO_MODE_V1
target_mass_code=1.92000000000000000e+06
final_mass_code=1.92000000000000000e+06
mean_mass_error=0.00000000000000000e+00
last_zero_mode_lambda=8.28397007500473634e-09
checkpoint_restart_provenance=RESTORED_AND_VALIDATED
elastic_solver_status=PASS_ELASTIC_WARM_START_RESIDUAL_V1
elastic_nonconverged_steps=0
elastic_mean_iterations=3
elastic_last_residual=9.44985475096211181e-07
segment_wall_seconds=1324.523
segment_mean_wall_seconds_per_step=0.364581
```

The segment and driver stderr logs remain empty. Its reported VTK paths are
suppression notices, not bulk-field output. The driver has resumed from this
checkpoint; this remains numerical-integrity evidence only and does not
establish a 48 h microstructure or transport result.

## Sixteenth registered one-hour checkpoint

The `step_58128` segment (approximately 22 h) independently reports:

```text
segment_status=PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_SEGMENT_V1
checkpoint=step_58128.chk
checkpoint_sha256=e3fc1b441116026be2c221947a785555802344bbe6833691d54b5809bb10630a
zero_mode_status=PASS_PF_CONSERVED_Y_ZERO_MODE_V1
target_mass_code=1.92000000000000000e+06
final_mass_code=1.92000000000000000e+06
mean_mass_error=0.00000000000000000e+00
last_zero_mode_lambda=4.16513251794058427e-09
checkpoint_restart_provenance=RESTORED_AND_VALIDATED
elastic_solver_status=PASS_ELASTIC_WARM_START_RESIDUAL_V1
elastic_nonconverged_steps=0
elastic_mean_iterations=3.019267823
elastic_last_residual=6.94076797800361597e-07
segment_wall_seconds=1329.691
segment_mean_wall_seconds_per_step=0.366004
```

The segment and driver stderr logs remain empty. Its reported VTK paths are
suppression notices, not bulk-field output. The driver has resumed from this
checkpoint; this remains numerical-integrity evidence only and does not
establish a 48 h microstructure or transport result.

## Seventeenth registered one-hour checkpoint

The `step_61761` segment (approximately 23 h) independently reports:

```text
segment_status=PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_SEGMENT_V1
checkpoint=step_61761.chk
checkpoint_sha256=c33f00d145b5f68e304a725baa8e7b5ae900b7376cc461ed2c3706587116ad7e
zero_mode_status=PASS_PF_CONSERVED_Y_ZERO_MODE_V1
target_mass_code=1.92000000000000000e+06
final_mass_code=1.92000000000000023e+06
mean_mass_error=3.63797880709171279e-18
last_zero_mode_lambda=2.07510872612288599e-09
checkpoint_restart_provenance=RESTORED_AND_VALIDATED
elastic_solver_status=PASS_ELASTIC_WARM_START_RESIDUAL_V1
elastic_nonconverged_steps=0
elastic_mean_iterations=3.005780347
elastic_last_residual=7.90848047465642520e-07
segment_wall_seconds=1325.222
segment_mean_wall_seconds_per_step=0.364773
```

## Eighteenth registered science checkpoint — 24 h

The `step_65393` segment is the frozen exact 24 h science endpoint. It
independently reports:

```text
segment_status=PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_SEGMENT_V1
checkpoint=step_65393.chk
checkpoint_sha256=217f1b0130e136f70b02e154723752bff6987ec91e28c8abcebdf1821a244760
zero_mode_status=PASS_PF_CONSERVED_Y_ZERO_MODE_V1
target_mass_code=1.92000000000000000e+06
final_mass_code=1.92000000000000000e+06
mean_mass_error=0.00000000000000000e+00
last_zero_mode_lambda=3.64914014783214586e-09
checkpoint_restart_provenance=RESTORED_AND_VALIDATED
elastic_solver_status=PASS_ELASTIC_WARM_START_RESIDUAL_V1
elastic_nonconverged_steps=0
elastic_mean_iterations=3.032764317
elastic_last_residual=6.53873172380587027e-07
segment_wall_seconds=1328.986
segment_mean_wall_seconds_per_step=0.365910
```

## Nineteenth registered one-hour checkpoint

The adjacent hourly `step_65394` segment preserves the ordinary one-hour
chain after the exact 24 h endpoint:

```text
segment_status=PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_SEGMENT_V1
checkpoint=step_65394.chk
checkpoint_sha256=48c1d0087da4d42d911529b65a21aae8999c55ae252dfaa93423e5f44383b497
zero_mode_status=PASS_PF_CONSERVED_Y_ZERO_MODE_V1
target_mass_code=1.92000000000000000e+06
final_mass_code=1.92000000000000000e+06
mean_mass_error=0.00000000000000000e+00
last_zero_mode_lambda=3.89509725248233663e-09
checkpoint_restart_provenance=RESTORED_AND_VALIDATED
elastic_solver_status=PASS_ELASTIC_WARM_START_RESIDUAL_V1
elastic_nonconverged_steps=0
elastic_mean_iterations=3
elastic_last_residual=6.49876992279269503e-07
segment_wall_seconds=22.545
segment_steps=1
```

All three segment and driver stderr logs remain empty. The driver resumed from
the chain's ordinary hourly checkpoint. These are numerical-integrity evidence
only and do not establish a 48 h microstructure or transport result.

## Twentieth registered one-hour checkpoint

The `step_69027` segment (approximately 25 h) independently reports:

```text
segment_status=PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_SEGMENT_V1
checkpoint=step_69027.chk
checkpoint_sha256=1f8c671199948889a2475c4e3315a4cb85982e04088b75f40e16a284c43b390d
zero_mode_status=PASS_PF_CONSERVED_Y_ZERO_MODE_V1
target_mass_code=1.92000000000000000e+06
final_mass_code=1.92000000000000047e+06
mean_mass_error=7.27595761418342557e-18
last_zero_mode_lambda=6.93841951952144339e-10
checkpoint_restart_provenance=RESTORED_AND_VALIDATED
elastic_solver_status=PASS_ELASTIC_WARM_START_RESIDUAL_V1
elastic_nonconverged_steps=0
elastic_mean_iterations=3
elastic_last_residual=6.35028316753186218e-07
segment_wall_seconds=1326.793
segment_mean_wall_seconds_per_step=0.365206
```

The segment and driver stderr logs remain empty. Its reported VTK paths are
suppression notices, not bulk-field output. The driver has resumed from this
checkpoint; this remains numerical-integrity evidence only and does not
establish a 48 h microstructure or transport result.

## Twenty-first registered one-hour checkpoint

The `step_72660` segment (approximately 26 h) independently reports:

```text
segment_status=PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_SEGMENT_V1
checkpoint=step_72660.chk
checkpoint_sha256=0b6ab43a9dc73ba88e0cebdfc8b7dc9a4291f43a858b03ebc715eab430adc002
zero_mode_status=PASS_PF_CONSERVED_Y_ZERO_MODE_V1
target_mass_code=1.92000000000000000e+06
final_mass_code=1.91999999999999977e+06
mean_mass_error=-3.63797880709171279e-18
last_zero_mode_lambda=1.18275928031617289e-09
checkpoint_restart_provenance=RESTORED_AND_VALIDATED
elastic_solver_status=PASS_ELASTIC_WARM_START_RESIDUAL_V1
elastic_nonconverged_steps=0
elastic_mean_iterations=3.004954583
elastic_last_residual=6.17810852028692624e-07
segment_wall_seconds=1328.840
segment_mean_wall_seconds_per_step=0.365769
```

The segment and driver stderr logs remain empty. Its reported VTK paths are
suppression notices, not bulk-field output. The driver has resumed from this
checkpoint; this remains numerical-integrity evidence only and does not
establish a 48 h microstructure or transport result.

## Twenty-second registered one-hour checkpoint

The `step_76293` segment (approximately 27 h) independently reports:

```text
segment_status=PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_SEGMENT_V1
checkpoint=step_76293.chk
checkpoint_sha256=c1201275d1fb020dbab37963e8ea4c3abe338fb71413a8c48c6e08f9765c1d91
zero_mode_status=PASS_PF_CONSERVED_Y_ZERO_MODE_V1
target_mass_code=1.92000000000000000e+06
final_mass_code=1.91999999999999977e+06
mean_mass_error=-3.63797880709171279e-18
last_zero_mode_lambda=1.44763867299962247e-09
checkpoint_restart_provenance=RESTORED_AND_VALIDATED
elastic_solver_status=PASS_ELASTIC_WARM_START_RESIDUAL_V1
elastic_nonconverged_steps=0
elastic_mean_iterations=3
elastic_last_residual=6.13050710389133641e-07
segment_wall_seconds=1327.788
segment_mean_wall_seconds_per_step=0.365480
```

The segment and driver stderr logs remain empty. Its reported VTK paths are
suppression notices, not bulk-field output. The driver has resumed from this
checkpoint; this remains numerical-integrity evidence only and does not
establish a 48 h microstructure or transport result.

## Twenty-third registered one-hour checkpoint

The `step_79926` segment (approximately 28 h) independently reports:

```text
segment_status=PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_SEGMENT_V1
checkpoint=step_79926.chk
checkpoint_sha256=afe05035e74953697f29f3334f9201e848d9dfb3fa5b11684ac31a34892da83f
zero_mode_status=PASS_PF_CONSERVED_Y_ZERO_MODE_V1
target_mass_code=1.92000000000000000e+06
final_mass_code=1.92000000000000000e+06
mean_mass_error=0.00000000000000000e+00
last_zero_mode_lambda=9.88462230709040779e-09
checkpoint_restart_provenance=RESTORED_AND_VALIDATED
elastic_solver_status=PASS_ELASTIC_WARM_START_RESIDUAL_V1
elastic_nonconverged_steps=0
elastic_mean_iterations=3
elastic_last_residual=4.70420898561883463e-07
segment_wall_seconds=1326.497
segment_mean_wall_seconds_per_step=0.365124
```

## Twenty-fourth registered one-hour checkpoint

The `step_83559` segment (approximately 29 h) independently reports:

```text
segment_status=PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_SEGMENT_V1
checkpoint=step_83559.chk
checkpoint_sha256=7980624a94467d376d49ae66376dddaa1993b9a2b30f2cfcd950f49c5b7342d2
zero_mode_status=PASS_PF_CONSERVED_Y_ZERO_MODE_V1
target_mass_code=1.92000000000000000e+06
final_mass_code=1.92000000000000000e+06
mean_mass_error=0.00000000000000000e+00
last_zero_mode_lambda=9.75214895656759957e-09
checkpoint_restart_provenance=RESTORED_AND_VALIDATED
elastic_solver_status=PASS_ELASTIC_WARM_START_RESIDUAL_V1
elastic_nonconverged_steps=0
elastic_mean_iterations=3
elastic_last_residual=6.74480988648326349e-07
segment_wall_seconds=1326.111
segment_mean_wall_seconds_per_step=0.365018
```

## Twenty-fifth registered one-hour checkpoint

The `step_87192` segment (approximately 30 h) independently reports:

```text
segment_status=PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_SEGMENT_V1
checkpoint=step_87192.chk
checkpoint_sha256=50920ca58382bd032a193b3c5dcadbf7b218a0c74534af73721d4d7090488d06
zero_mode_status=PASS_PF_CONSERVED_Y_ZERO_MODE_V1
target_mass_code=1.92000000000000000e+06
final_mass_code=1.91999999999999977e+06
mean_mass_error=-3.63797880709171279e-18
last_zero_mode_lambda=7.77750696572297590e-10
checkpoint_restart_provenance=RESTORED_AND_VALIDATED
elastic_solver_status=PASS_ELASTIC_WARM_START_RESIDUAL_V1
elastic_nonconverged_steps=0
elastic_mean_iterations=3
elastic_last_residual=5.58767342317890684e-07
segment_wall_seconds=1326.343
segment_mean_wall_seconds_per_step=0.365082
```

The segment and driver stderr logs remain empty. Their reported VTK paths are
suppression notices, not bulk-field output. The driver resumed from the last
checkpoint; these remain numerical-integrity evidence only and do not
establish a 48 h microstructure or transport result.

## Twenty-sixth registered one-hour checkpoint

The `step_90825` segment (approximately 31 h) independently reports:

```text
segment_status=PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_SEGMENT_V1
checkpoint=step_90825.chk
checkpoint_sha256=b07ac10e536ea205015573dca20282585c55d7aff30a1fe80d96321ce0f6af6a
zero_mode_status=PASS_PF_CONSERVED_Y_ZERO_MODE_V1
target_mass_code=1.92000000000000000e+06
final_mass_code=1.92000000000000000e+06
mean_mass_error=0.00000000000000000e+00
last_zero_mode_lambda=1.30183524411109767e-08
checkpoint_restart_provenance=RESTORED_AND_VALIDATED
elastic_solver_status=PASS_ELASTIC_WARM_START_RESIDUAL_V1
elastic_nonconverged_steps=0
elastic_mean_iterations=3
elastic_last_residual=5.57415815526050696e-07
segment_wall_seconds=1326.513
segment_mean_wall_seconds_per_step=0.365129
```

The segment and driver stderr logs remain empty. Its reported VTK paths are
suppression notices, not bulk-field output. The driver has resumed from this
checkpoint; this remains numerical-integrity evidence only and does not
establish a 48 h microstructure or transport result.

## Twenty-seventh registered one-hour checkpoint

The `step_94458` segment (approximately 32 h) independently reports:

```text
segment_status=PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_SEGMENT_V1
checkpoint=step_94458.chk
checkpoint_sha256=8b6c40dfa6c3872acbfa572c32bb660e5d2b40bd9ca6765beaf3a5c939bc3c7b
zero_mode_status=PASS_PF_CONSERVED_Y_ZERO_MODE_V1
target_mass_code=1.92000000000000000e+06
final_mass_code=1.92000000000000000e+06
mean_mass_error=0.00000000000000000e+00
last_zero_mode_lambda=3.56625738490573437e-09
checkpoint_restart_provenance=RESTORED_AND_VALIDATED
elastic_solver_status=PASS_ELASTIC_WARM_START_RESIDUAL_V1
elastic_nonconverged_steps=0
elastic_mean_iterations=3
elastic_last_residual=2.72329941864401677e-07
segment_wall_seconds=1324.916
segment_mean_wall_seconds_per_step=0.364689
```

The segment and driver stderr logs remain empty. Its reported VTK paths are
suppression notices, not bulk-field output. The driver has resumed from this
checkpoint; this remains numerical-integrity evidence only and does not
establish a 48 h microstructure or transport result.

## Completion-audit schedule correction

Before the production reached its postprocessing phase, the read-only
completion audit was corrected to derive the expected 44 checkpoints from the
campaign's hourly cadence **and** its frozen registered science steps. The
previous version would have rejected the valid chain because it omitted the
additional exact physical-time endpoints `65393` (24 h) and `108989` (36 h),
each intentionally adjacent to the hourly integer-step endpoints `65394` and
`108990`.

```text
analysis_script=audit_pf_400cube_fixed_xb03_elastic_psd_sign_pilot_v1.py
analysis_script_sha256=bb09e883cf50d44090b16dda4320c2927d11dd0091eb6a766816b4530f17a051
regression_status=PASS_400CUBE_SCHEDULE_AUDIT_REGRESSION
expected_endpoint_count=44
```

This changes neither the running PF binary, parameter file, fixture, output
schedule, checkpoint bytes, nor any physical model term. The production driver
will include the corrected script hash in its final `analysis_hashes.sha256`.
