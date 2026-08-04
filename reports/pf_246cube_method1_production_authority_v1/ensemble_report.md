# Experiment-matrix-anchored Method-1 A/B/C no-dislocation transport ensemble

## Status and authority boundary

```text
status=PASS_PF_FULL_PSD_NO_DISLOCATION_TRANSPORT_ENSEMBLE_V1
replicates=A,B,C
registered_ages_h=6,12,18,24,36,48
A_N=1.5
dislocation_mode=DISLOCATION_OFF
S11_rate=0
S13_rate=0
yu_refit_scale_used=false
absolute_experimental_kappa_claim=false
```

Each selected trajectory is the quarter-nm Method-1,
experiment-matrix-anchored 246^3 production field from job 73364, 73365, or
73366. The legacy production drivers retained their original low-threshold
`BLOCKED` root status; those immutable roots were not edited. Each selected
authority is instead a separate overlay that requires all 44 solver segments,
all 44 remote checkpoint SHA-256 values, the frozen fixture/campaign/input
ledger, and the replacement three-threshold hourly merge-aware audit. Every
one of those gates passes for A, B, and C.

The transport calculations use only the resolved physical component PSD at
the six registered science ages. C's two persistent strong-core merges are
represented as physical connected components, not as extra independent
particles. These are conditional PF-PSD, no-dislocation lattice-transport
results, not a reproduction of absolute experimental thermal conductivity.

## Full-PSD ensemble at 48 h

Values are mean +/- sample standard deviation over the three hash-pinned
replicates, in W m^-1 K^-1. `fixed matrix` holds each replicate's 6 h
far-field Ag concentration; `PF matrix` uses the time-varying far-field Ag.

| T (K) | kappa fixed matrix | kappa PF matrix | delta kappa PSD | delta kappa PSD + matrix |
|---:|---:|---:|---:|---:|
| 300 | 2.284604 +/- 0.001512 | 2.279808 +/- 0.006337 | -0.001867 +/- 0.001507 | -0.006662 +/- 0.006330 |
| 400 | 1.786014 +/- 0.000623 | 1.783033 +/- 0.003625 | -0.002080 +/- 0.000620 | -0.005061 +/- 0.003620 |
| 600 | 1.243672 +/- 0.000140 | 1.242200 +/- 0.001616 | -0.001518 +/- 0.000139 | -0.002991 +/- 0.001614 |

The complete 6--48 h by-temperature table, including min/max, is
`ensemble/ensemble_kappa_time_temperature.csv`.

## Descriptor sufficiency against direct full PSD

The direct full-PSD particle sum is the reference. MAPE below is aggregated
over A/B/C, all six registered ages and seven temperatures (126 cells per
matrix mode). `M6` alone is its Rayleigh-limit closure; `Sv+M6` is the
two-moment reconstructed population.

| Descriptor | fixed-matrix MAPE | PF-matrix MAPE |
|---|---:|---:|
| full PSD | 0 | 0 |
| Nv + mean R | 0.3281% | 0.3293% |
| Sv | 4.4431% | 4.4593% |
| M6 Rayleigh limit | 75.4356% | 75.3539% |
| Sv + M6 reconstruction | 0.08955% | 0.08992% |

Thus `Sv` alone and `M6` alone are insufficient on this 6--48 h ensemble.
`Sv+M6` is the best tested compressed representation here, but full PSD
remains the production authority because moment closures are not unique PSDs.

## Verification and provenance

- Every authority overlay manifest verifies locally by SHA-256.
- The existing authority adapter verified all transport output hashes before
  ensemble assembly; an independent post-assembly hash check also passed.
- Each transport manifest preserves the exact production PASS status, all-true
  production gates, 44 checkpoint hashes, source binary/parameter/analysis
  hashes, fixture, campaign, raw registered PSD and observables.

```text
authority_assembly_script_sha256=78ed4c65603e62c5f959f05f089a4c491a968ab3b70f2d5dfaeb45174c502513
production_adapter_script_sha256=6e85ecb29f767fe96fe2ff4e9262ddf547629a6d538cc880d4d6d829a021104d
ensemble_assembler_script_sha256=84763345be6726b419e0cccf093880f328f1854f91ba5a6ecf3c127692728ce0
authority_selection_sha256=8574b0a0c091f936a4fdac8ab4258c9816627f690f148a74a1bdf92d2129da56
ensemble_manifest_sha256=4c726fd8aaac155d4fc746c68b8f6091bb4d31527a5ae008178a8e6eceac9cd7
ensemble_kappa_sha256=3d3df325569ba8886d15e37802afd9c1b93a6a5c873a41eabc75ff3376e14f62
ensemble_descriptor_sha256=0168864bb47db3d0fe9a2ca549710a0813f47f75d5e3f47f282c738eb19f6e45
```
