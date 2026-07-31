# Elastic target-profile cluster timestep refinement

## Decision

The complete eight-radius ladder passed the registered cluster-only
timestep-refinement contract:

```text
status=PASS_ELASTIC_TARGET_PROFILE_DT_AND_DEVICE_REFINEMENT_V1
scientific_interpretation=PASS_CLUSTER_TIMESTEP_REFINEMENT
candidate_minimize_dt=0.025
reference_minimize_dt=0.05
profile_count=8
radii_nm=8.0 8.5 9.0 9.5 10.0 10.5 11.0 11.5
```

The analyzer schema retains the historical word `DEVICE`, but this comparison
used the same cluster source tree and the same cluster binary. It is a
timestep-refinement result, not a requirement for workstation/cluster
duplication.

## Frozen provenance

| Item | SHA-256 |
|---|---|
| Selected `dt=0.025` library manifest | `58803a8bc6679b823e45e7a7b85df16ae68efa55338d52d4c4151b414a5ef0fe` |
| Selected `dt=0.025` selection provenance | `56c44d8f72b27bb462dffe89b59cb2fcb2ff0bf8807dec9d9cac31d2bd7fcbe3` |
| Reference `dt=0.05` library manifest | `fbd5bc65e7d477475c1f4ab572bfe110915921ded9f3075c18a4d7f9ae0937b1` |
| Refinement audit | `81755a023dbbedd9b7c3a27dfcc54b92f6245f47ea4edc79976fdb93c36ff490` |
| Common source tree | `f7855699addf98f9d5aed03af876d851d524fb62c98f5a48df78d1fe561a2a75` |
| Cluster binary | `55cf917df94fcf01373d62f54f9ab99715975863ad5dcfb95baa8a517460cda1` |

The source commit recorded by both libraries is
`5bb0db2bb3321b70cb5f254ffd9ab738a3b3611d`.

## Radius-resolved comparison

| R (nm) | Canonical-field L1 | \(\phi\) normalized L1 | Mean \(|\Delta x_B|\) | Max axis change | Pseudo-time difference / allowance |
|---:|---:|---:|---:|---:|---:|
| 8.0 | \(1.4563\times10^{-5}\) | \(1.0171\times10^{-4}\) | \(8.8639\times10^{-8}\) | \(1.4221\times10^{-5}\) | 0.475 / 7.5 |
| 8.5 | \(2.4198\times10^{-5}\) | \(1.1860\times10^{-4}\) | \(1.4818\times10^{-7}\) | \(2.3208\times10^{-5}\) | 0.475 / 7.5 |
| 9.0 | \(3.2347\times10^{-5}\) | \(1.1650\times10^{-4}\) | \(1.9337\times10^{-7}\) | \(3.1949\times10^{-5}\) | 0.475 / 7.5 |
| 9.5 | \(3.5247\times10^{-5}\) | \(1.1250\times10^{-4}\) | \(2.0663\times10^{-7}\) | \(3.5103\times10^{-5}\) | 0.475 / 7.5 |
| 10.0 | \(4.9764\times10^{-5}\) | \(1.2828\times10^{-4}\) | \(2.8246\times10^{-7}\) | \(5.0241\times10^{-5}\) | 0.475 / 7.5 |
| 10.5 | \(6.0959\times10^{-5}\) | \(1.3553\times10^{-4}\) | \(3.3524\times10^{-7}\) | \(6.2377\times10^{-5}\) | 0.400 / 7.5 |
| 11.0 | \(7.2154\times10^{-5}\) | \(1.4761\times10^{-4}\) | \(3.8773\times10^{-7}\) | \(7.3542\times10^{-5}\) | 2.125 / 7.5 |
| 11.5 | \(8.6795\times10^{-5}\) | \(1.5619\times10^{-4}\) | \(4.5697\times10^{-7}\) | \(8.6316\times10^{-5}\) | 6.275 / 7.5 |

All field, shape, centroid, far-field composition, minimum relaxation-depth,
and pseudo-time alignment gates pass. The largest canonical-field difference
is \(8.68\times10^{-5}\), below the \(10^{-4}\) hard limit.

## Preserved failures and contract interpretation

The original full `dt=0.025` root stopped fail-closed at R=11.5 nm because
the projected update entered a stable projection-cycle floor just above the
old \(4\times10^{-5}\) rate threshold. KKT, energy, volume, composition-rate,
and mass gates had already passed. The stopping contract was revised
uniformly to \(5\times10^{-5}\) for the energy-plateau branch, and the
R=11.5 endpoint was regenerated with an extended iteration budget.

An earlier strict comparison also failed only because it required nearly
identical stopping pseudo-times. The final contract allows the larger of one
registered consecutive-step window or 2.5% of the common minimum relaxation
depth. This only aligns equivalent projection-cycle endpoints; none of the
field, mass, KKT, composition, or shape thresholds was relaxed.

The earlier roots and reports remain preserved and are not promoted as
production evidence.

## Qualified roots

- Selected `dt=0.025`:
  `/data/home/luozhiheng/tmp/pf_elastic_target_profile_cluster_selected_minpt300_dt0p025_v16_20260730`
- Reference `dt=0.05`:
  `/data/home/luozhiheng/tmp/pf_elastic_target_profile_cluster_selected_minpt300_dt0p05_v18_20260730`

Compact manifests and audits are frozen under
`evidence/cluster_selected_dt0p025_v16` and
`evidence/cluster_selected_dt0p05_v18`.
