# 400-cube fixed-xB=0.03 elastic PSD-sign pilot — V3 submission registry

**Status at 2026-08-02T06:26:26Z:**
`PREPARATION_RUNNING_STATIC_LIBRARY_AND_FIXTURE_GATES`

This is a user-authorized direct 400 nm-cube elastic pilot.  It is not a
box-convergence study, an ensemble qualification, or an absolute experimental
reconstruction.  The scope remains post-nucleation PF-only resolved-beta
evolution: GP, external sources, and new-particle insertion are disabled.

## Why V3 was submitted

The preserved V1 and V2 preparations used a 96-cube native profile window.
R17 did not reach the unchanged full-model energy/convergence contract in
that window, so neither preparation produced a fixture or a production
output.  No V1/V2 field is reused here.

An isolated R17 native 160-cube probe was then completed before V3:

```text
probe_root=/data/home/luozhiheng/tmp/pf_400cube_R17_grid160_profile_probe_v1_20260802
probe_job_id=73796
probe_state=COMPLETED
probe_status=PASS_PF_ELASTIC_TARGET_PROFILE_LIBRARY_V1
grid=160x160x160; dx_nm=1; lambda_sm_nm=4
converged_iter=31214; trigger=condB
rms_res=4.28760793867336324e-04
rms_dphi=3.03289330723283133e-05
rms_dY=5.24522327650188961e-08
energy_diff_rel=4.24444609866541993e-09
volume_error_relative=1.81580993587286537e-11
mass_error_relative=1.57862147909343349e-16
probe_library_manifest_sha256=d38e8a13607e6378b2acf5c00feee9b219e97de110208e0f8f0919dc8e162001
```

Only the isolated numerical profile window changed between the failed 96-cube
attempt and this probe.  The temperature, (dx), interface width,
thermodynamics, kinetics, elasticity, far-field composition, minimizer time
step, mass contract, and acceptance thresholds did not change.

## V3 preparation submission

```text
preparation_job_id=73798
queue_state_at_submission=RUNNING
partition=gpu_uvip
qos=gpu_uvip
resources=1 GPU, 4 CPUs, mem=0, 12 h
submitted_source=/data/home/luozhiheng/tmp/codex_pf_400cube_fixed_xb03_elastic_psd_sign_v3_20260802
preparation_root=/data/home/luozhiheng/tmp/pf_400cube_fixed_xb03_elastic_psd_sign_prepare_v3_20260802
reserved_nonoverwriting_production_root=/data/home/luozhiheng/tmp/pf_400cube_6h48h_fixed_xb03_elastic_psd_sign_pilot_v3_20260802
profile_grid=160x160x160
registered_native_radii_nm=17.0,17.5,18.0,18.5,19.0,19.5,20.0
fixture_spec=fixture_spec_native160_v1.json
fixture_spec_sha256=a87daffc3816bebcff7c8735aace3e00ab40100649383fe4f7a3c2c958994410
prepare_sbatch_sha256=8fa8f3bea1ddca0dba60451d544ec1cc8dc2a378d4549a5ffef3910dd551f4bd
production_sbatch_sha256=ea490d4f6bcb17ab9970dd84ea3dba0a71981a3e4dbb499a861b106e08c146b0
```

The preparation job must first generate and pass all seven native profiles,
then materialize and statically audit the 400-cube fixture, and finally run a
zero-macrostep production-binary preflight.  Only then does it submit the
separate 48 h production allocation and its read-only tail job.  Consequently
there is no production job ID, binary hash, parameter hash, fixture hash, or
checkpoint-chain hash to register yet.

The eventual production allocation is frozen to one GPU, four CPUs, `mem=0`,
`gpu_uvip/gpu_uvip`, 48 h, `dt_code=0.02`, `dt_physical_s=0.9909260953431841`,
and `final_step=152585`; it must not emit bulk VTK fields.

