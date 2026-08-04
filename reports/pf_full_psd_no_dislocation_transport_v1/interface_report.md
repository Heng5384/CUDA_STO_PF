# PF full-PSD no-dislocation lattice transport V1

## Decision

```text
PASS_PF_FULL_PSD_NO_DISLOCATION_TRANSPORT_INTERFACE_V1
```

This status qualifies the PF-to-lattice-transport interface, its numerical
implementation, historical trajectory smoke, provenance closure, and future
A/B/C fail-closed authority selector. It does **not** claim absolute
experimental lattice-thermal-conductivity reproduction.

## Frozen scientific contract

- `A_N = 1.5`.
- SI S11 dislocation-core rate is identically zero.
- SI S13 dislocation-strain rate is identically zero.
- The Yu `s_dis = 0.1172768` refit diagnostic is not read or used.
- The Yu small/big single-radius precipitate populations are replaced by the
  resolved PF particle population.
- The remaining host baseline is the public Yu 48 h non-particle parameter
  block: lattice constant 6.445 Å, grain size 12.1 µm, average sound velocity
  1770 m/s, Debye temperature 136 K, and Grüneisen parameter 1.96.
- PF matrix `xAg` changes only the point-defect parameter `Gamma`. No
  composition-dependent lattice, grain, sound, or Debye law is invented.

The full-PSD precipitate rate is

\[
\tau_{\rm Pre}^{-1}(\omega,t)
=
\frac{v}{V_{\rm box}}
\sum_i
\left[
\sigma_S(R_i)^{-1}+\sigma_l(R_i,\omega)^{-1}
\right]^{-1}.
\]

Each resolved PF particle carries the exact number-density weight
`1/V_box`. The interface stores the particle radii, `Nv`, spherical-equivalent
`Sv`, `M6`, matrix `xAg/xB`, age, step, box volume, and complete upstream
source/parameter/binary/fixture/analysis provenance.

## Scientific outputs

For the registered 6 h reference snapshot,

\[
\Delta\kappa_{\rm PSD}(T,t)
=
\kappa_L^{\rm no-dis}(T,t;x_{Ag}^{6h})
-\kappa_L^{\rm no-dis}(T,6h;x_{Ag}^{6h}),
\]

and

\[
\Delta\kappa_{\rm PSD+matrix}(T,t)
=
\kappa_L^{\rm no-dis}(T,t;x_{Ag}(t))
-\kappa_L^{\rm no-dis}(T,6h;x_{Ag}^{6h}).
\]

These are conditional resolved-PF contributions. Unresolved 1--5 nm objects,
external dislocations, and any unmodelled defect populations are outside this
V1 result.

## Historical 6--48 h interface smoke

The smoke consumes 43 registered snapshots and 777 resolved-particle rows.
The historical trajectory is not promoted to the future A/B/C ensemble. Its
overall snapshot PSD is used; no independent post-merge particle-lineage
claim is made. The old strict merge block and the later merge-aware
supplemental evidence are both preserved in the provenance manifest.

The historical matrix observation is `h(phi)<0.005`, explicitly labelled as a
smoke-only observation contract. Future experiment-matrix-anchored A/B/C
results must use their registered far-field observation contract.

Endpoint examples are:

| T (K) | 6 h κ no-dis | 48 h fixed-matrix κ | 48 h time-varying-matrix κ | Δκ PSD | Δκ PSD+matrix |
|---:|---:|---:|---:|---:|---:|
| 300 | 2.286555 | 2.283500 | 2.277913 | −0.003055 | −0.008642 |
| 400 | 1.788146 | 1.785531 | 1.782060 | −0.002616 | −0.006087 |
| 500 | 1.468068 | 1.465977 | 1.463613 | −0.002091 | −0.004455 |
| 600 | 1.245217 | 1.243547 | 1.241834 | −0.001670 | −0.003383 |

Units are W m⁻¹ K⁻¹. These numbers are interface smoke outputs, not absolute
experimental predictions.

## Descriptor sufficiency diagnostic

Errors below use all 43 historical snapshots and seven temperatures, with
full PSD as the reference.

| Descriptor model | Fixed-matrix MAPE | Time-varying-matrix MAPE | Maximum error |
|---|---:|---:|---:|
| `Nv + mean R` monodisperse | 0.278% | 0.279% | 0.846% |
| `Sv` geometric limit | 4.108% | 4.131% | 8.793% |
| `M6` Rayleigh limit | 78.179% | 78.059% | 88.467% |
| `Sv + M6` moment reconstruction | 0.0835% | 0.0838% | 0.175% |
| full PSD | 0 | 0 | 0 |

`Sv` and `M6` alone are deliberately evaluated as their respective geometric
and Rayleigh limiting rates. The excellent historical `Sv+M6` result is a
property of this smoke trajectory and must be re-evaluated on the independent
A/B/C ensemble; it is not yet a universal sufficiency claim.

## Qualification gates

| Gate | Result |
|---|---|
| frozen `A_N`, S11/S13 off, no refit scale | PASS |
| monodisperse degeneration | PASS |
| vectorized full PSD versus scalar direct sum | PASS |
| PSD bin refinement | PASS |
| `Sv+M6` moment reconstruction | PASS |
| 256/512-point and adaptive Debye integration | PASS |
| historical 43-snapshot 6--48 h smoke | PASS |
| byte-deterministic regenerated outputs | PASS |
| A/B/C unique-authority selector, positive and fail-closed tests | PASS |

Numerical maxima:

- direct-sum relative difference: `3.25e-16`;
- source `Nv/Sv/M6/mean-R` closure: `8.49e-16`;
- 256/512-point Debye difference: `7.32e-15`;
- Gauss/adaptive Debye difference: `2.76e-15`;
- 0.125 nm PSD-bin relative κ error in the broad synthetic test: `1.63e-6`;
- two regenerated output trees have identical per-file SHA-256 hashes.

## 246³ production-PASS ingress

The historical smoke and the registered 246³ production audit use different
field names and authority contracts. A dedicated fail-closed adapter now maps
only an exact complete production PASS into the transport snapshot schema. It
requires all production gates (including merge-aware lineage) to be true,
exactly the six registered 6/12/18/24/36/48 h snapshots, full-PSD count and
`Nv/Sv/M6/mean-R` closure, an experiment-matrix-anchored trajectory class,
the complete 44-checkpoint hash chain, and complete
source/binary/parameter/fixture/analysis hashes.

The adapter was qualified with a production-schema synthetic fixture derived
from the already-qualified historical smoke. Its maximum descriptor closure
error was `8.79e-16`; two independent conversions produced identical hashes
for all six transport outputs. Negative tests confirmed fail-closed behavior
for a non-anchored trajectory, a failed merge-aware gate, an incomplete PSD,
an incomplete checkpoint chain, and a fixture-hash mismatch. The hardened
ensemble selector additionally rejects a missing exact production-PASS
provenance record and any missing or duplicate A/B/C × age × temperature ×
matrix-mode × descriptor cell.

```text
production_ingress_status=PASS_PF_246CUBE_NO_DISLOCATION_TRANSPORT_AUTHORITY_ADAPTER_V1
production_A_B_C_data_used=false
running_PF_jobs_modified=false
```

## A/B/C production boundary

No running PF A/B/C job, input, checkpoint, output, scheduler state, or source
identity was changed or submitted by this work.

The ensemble assembler remains fail-closed until the experiment-matrix-anchored
A/B/C runs finish and each replicate supplies exactly one complete PASS
authority. It rejects zero or multiple selections, incomplete trajectories,
missing 6/12/18/24/36/48 h snapshots, wrong hashes, duplicate trajectory
identities, refitted/dislocation-enabled contracts, and non-experiment-matrix
trajectory classes.

The eventual ensemble will report mean, sample standard deviation, minimum,
and maximum of `kappa_L_no_dis`, `delta_kappa_PSD`, and
`delta_kappa_PSD_plus_matrix`, with one and only one authority per replicate.
The current Method-1 A/B/C candidate identities are frozen in
`data/qualification/pf_full_psd_no_dislocation_transport_v1/abc_quarter_nm_method1_authority_manifest.pending.json`.
It intentionally contains zero selections and is verified to fail closed until
all three production authorities exist. The requirement-by-requirement state
is maintained in `completion_audit.md`.

## Frozen provenance

```text
transport_interface_script_sha256=a42f933c43d0adbd4fc9a4d86a7b74bc08d4efe64517cc3fbf1ba177b62dcf66
qualification_script_sha256=8b289bf18c15529976b81e69d285ef76db5b443ac7d0f46364259f6ac0f2f183
ensemble_assembler_script_sha256=84763345be6726b419e0cccf093880f328f1854f91ba5a6ecf3c127692728ce0
transport_parameter_contract_sha256=d16e948dd34403129b2deecb90efbea937251e23ae9c128c353b6f66bbee1ae0
yu_public_parameter_config_sha256=163562fa19f0cbcf3731f181ecae6c1d34f4b5952cdab9a81a5aa3eabd9f1c8f
historical_transport_manifest_sha256=9ad2b0081832b073f84ad7404d4b2b0f77d56a194f495a3593aab95991d32179
qualification_json_sha256=abdd85f9a6e59369674533b980c1d4215901a4c2c22dbea76453b8d5b7a94d32
production_adapter_script_sha256=6e85ecb29f767fe96fe2ff4e9262ddf547629a6d538cc880d4d6d829a021104d
production_adapter_qualification_script_sha256=17a160d94c076379d5563b9e9b587c9182b845c1e7c679a657b7100b28c8f7f6
production_adapter_qualification_json_sha256=fc3356b0b3c3d3e96cf9d8f4119a97a1e520ee413e09b60052ab7b3997ad8887
```
