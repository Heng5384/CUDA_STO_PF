# Workstation elastic target-profile implementation proof

## Historical implementation decision

The complete registered T380 radius ladder passed on `workstation-tail`:

```text
runner_status=PASS_PF_ELASTIC_TARGET_PROFILE_LIBRARY_V1
grid=96x96x96
minimize_dt=0.1
radii_nm=8.0 8.5 9.0 9.5 10.0 10.5 11.0 11.5
elastic_enabled=true
gp_enabled=false
```

All eight cases used the same source-tree hash
`f22202bdde8bd93732fa43fe95f68c8ca5816723e24204c8569e46d88bad2c90`
and binary hash
`23d4d365de19c26670ed5d64f9f732ae71b555d2ef3ac57f7f5452467b7902d0`.
Every case had empty runtime stderr.

This `minimize_dt=0.1` ladder proves the implementation and materialization
path on the workstation. It is not the final selected production library.
The production entries are the cluster `minimize_dt=0.025` ladder documented
in `cluster_dt_refinement.md`; the workstation subsequently loaded three
hash-verified selected entries for dynamic/restart/dt qualification. This
complementary split avoids duplicating the full selected ladder solely for a
device comparison.

## Radius-resolved results

| Target R (nm) | Actual R (nm) | Major/minor axis ratio | Far-field \(x_B^\alpha\) | Final projected KKT RMS | Converged iteration | Mass relative error |
|---:|---:|---:|---:|---:|---:|---:|
| 8.0 | 8.0000000002 | 1.348821 | 0.006308102 | 4.491e-4 | 1071 | 1.18e-16 |
| 8.5 | 8.5000000005 | 1.370476 | 0.006313707 | 4.774e-4 | 1460 | 1.12e-16 |
| 9.0 | 9.0000000006 | 1.390951 | 0.006319313 | 5.232e-4 | 1125 | 2.11e-16 |
| 9.5 | 9.5000000005 | 1.410637 | 0.006324901 | 5.291e-4 | 1089 | 1.98e-16 |
| 10.0 | 10.0000000008 | 1.432477 | 0.006330550 | 5.892e-4 | 1790 | 1.86e-16 |
| 10.5 | 10.5000000009 | 1.452526 | 0.006336206 | 6.068e-4 | 2324 | 1.75e-16 |
| 11.0 | 11.0000000010 | 1.473723 | 0.006341874 | 6.401e-4 | 2861 | 1.63e-16 |
| 11.5 | 11.5000000011 | 1.494348 | 0.006347575 | 6.652e-4 | 3236 | 4.57e-16 |

The monotonic increase in axis ratio is a direct result of constrained
elastic relaxation.  The analytic spherical field is only the deterministic
initial guess and is not the materialized profile.

## Frozen evidence

- Workstation run root:
  `/home/zhiheng/tmp/pf_elastic_target_profile_library_workstation_v3_20260730`
- Complete local raw-field mirror:
  `/Users/heng/tmp/pf_elastic_target_profile_library_workstation_v3_20260730`
- Frozen manifest:
  `evidence/workstation_library_manifest.json`
- Library-manifest SHA-256:
  `961df153c1404fe08ab1ad04d5bdba1841858419571d9ca842fcad0671ec33bc`

The raw-field mirror is deliberately outside Git because it is approximately
395 MB.  Git contains the contracts, source, scripts, hashes, and compact
manifests.
