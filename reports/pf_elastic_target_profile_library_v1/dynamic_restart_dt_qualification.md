# Elastic profile dynamic handoff, restart, and timestep qualification

## Scope

Three pre-registered profiles span the small, middle, and large ends of the
V1 ladder: 8.0, 9.5, and 11.5 nm.  Each was loaded through the production
raw-field path with elasticity and the Ji-Chen conserved-Y zero mode enabled,
while every GP, source, and new-nucleation path remained disabled.

All three profiles are exact entries from the selected cluster `dt=0.025`
library with manifest SHA-256
`58803a8bc6679b823e45e7a7b85df16ae68efa55338d52d4c4151b414a5ef0fe`.
The entries were hash-verified after transfer to the workstation; they were
not regenerated on the workstation.

Each case contains:

1. a one-production-step handoff probe at `dt_code=0.02`;
2. a 64-step continuous run;
3. a 32+restart+32-step run;
4. a 128-step `dt_code=0.01` run to the same physical endpoint.

The one-step probe tests whether the minimized anisotropic shape survives the
runtime handoff.  Growth or shrinkage over the 64-step physical continuation
is not mislabeled as initialization relaxation.

## Results

| R (nm) | One-step normalized-axis change | One-step axis-ratio change | One-step h-volume change | dt phi normalized L1 | dt axis difference | dt mean \(|\Delta x_B|\) | Mass relative error | Restart |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 8.0 | 9.993e-4 | 6.695e-4 | 1.803e-2 | 4.675e-3 | 1.368e-3 | 4.379e-5 | 4.246e-15 | bytewise |
| 9.5 | 6.019e-4 | 6.031e-4 | 1.196e-2 | 2.747e-3 | 8.357e-4 | 4.163e-5 | 8.733e-15 | bytewise |
| 11.5 | 3.761e-4 | 5.124e-4 | 7.514e-3 | 1.523e-3 | 5.138e-4 | 3.912e-5 | 4.568e-16 | bytewise |

All three cases pass the hard qualification:

```text
dynamic_profile_status=PASS_ELASTIC_TARGET_PROFILE_DYNAMIC_RESTART_AND_DT_V1
initial_profile_handoff_status=PASS
restart_bytewise=true
zero_mode_status=PASS
gp_paths_enabled=false
```

An independent periodic six-neighbour connected-component audit at
`h(phi)>1e-4` found exactly one particle in the initial, one-step,
continuous, restart, and refined fields for every radius.  Initial-to-final
support overlap exceeded the pre-registered 90% floor, and continuous versus
restart `phi` was exact.  Therefore no merge, split, duplicate, or lost
identity is hidden by the global shape metrics.

The `dt` versus `dt/2` composition difference is below the registered hard
limit \(5\times10^{-5}\), but not below the preferred
\(2\times10^{-5}\) grade.  Therefore the library passes the dynamic numerical
gate without receiving the preferred composition-refinement grade.

Continuous and restarted final checkpoint hashes are identical for every
case:

- R=8.0 nm: `6ca75910f17cf0a58f3ac404343a52a59d14d32531ff4dd0a6b0678ffd59b175`;
- R=9.5 nm: `35703a80f207ca058aa38e321d9c10e00fd385bfa48a91c9eb2875a0ee4f5386`;
- R=11.5 nm: `da9e197b671d18ebcc1df673b06aea496bc2909413db2b1e6281a9cb1bd4159c`.

## Preserved first failure

The original R=8 nm v1 analyzer compared the initial profile against the
field after approximately 63.4 physical seconds and incorrectly treated
ordinary radius evolution as an insertion jump.  That non-overwritten run is
preserved at:

`/home/zhiheng/tmp/pf_elastic_target_profile_dynamic_R8_v1_20260730`

It already proved bytewise restart and exact mass closure.  The v2 contract
corrects the observation, rather than deleting or relabeling the v1 result.

## Qualified evidence roots

- R=8.0 nm:
  `/home/zhiheng/tmp/pf_elastic_target_profile_selected_v16_dynamic_R8p0_v1_20260730`
- R=9.5 nm:
  `/home/zhiheng/tmp/pf_elastic_target_profile_selected_v16_dynamic_R9p5_v1_20260730`
- R=11.5 nm:
  `/home/zhiheng/tmp/pf_elastic_target_profile_selected_v16_dynamic_R11p5_v1_20260730`

Machine-readable audits are frozen under `evidence/selected_dynamic_R8`,
`evidence/selected_dynamic_R9p5`, and
`evidence/selected_dynamic_R11p5`.
