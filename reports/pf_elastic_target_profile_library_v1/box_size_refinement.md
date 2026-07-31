# Elastic target-profile finite-box qualification

## Scope

The largest and most anisotropic registered particle, \(R=11.5\) nm, was
used as the fail-closed finite-box probe.  Four isolated periodic fixed-cell
elastic minimizations were generated:

```text
96^3, 128^3, 160^3, 192^3
dx=1 nm
T=380 °C
lambda_sm=4 nm
orientation=variant_100_identity
eigenstrain=(0.046,-0.022,-0.017,0,0,0)
```

All four runs used the same thermodynamics, constrained \(h(\phi)\) volume,
canonical mass ledger, source-tree SHA-256
`f22202bdde8bd93732fa43fe95f68c8ca5816723e24204c8569e46d88bad2c90`,
and binary SHA-256
`23d4d365de19c26670ed5d64f9f732ae71b555d2ef3ac57f7f5452467b7902d0`.
Every run passed its direct convergence, volume, exact-mass, and elastic
energy-provenance gates.

## Pairwise convergence

The comparison uses a centered periodic 64 nm window for local fields and
the box-integrated elastic energy

\[
F_{\rm el}^{\rm int}
=F_{\rm el}^{\rm mean}N_{\rm voxel}dx^3.
\]

Comparing box-mean energy density directly would be invalid because the box
volumes differ.

| pair | axis-ratio change | maximum semi-axis change | local \(\phi\) L1 | integrated elastic-energy change | far-subtracted local \(x_B^\alpha\) | absolute far \(x_B^\alpha\) |
|---|---:|---:|---:|---:|---:|---:|
| 96→128 | 0.2678% | 0.1579% | 0.2354% | 2.3190% | \(2.060\times10^{-6}\) | \(7.441\times10^{-5}\) |
| 128→160 | 0.1529% | 0.0917% | 0.1271% | 0.9308% | \(9.888\times10^{-7}\) | \(2.634\times10^{-5}\) |
| 160→192 | 0.0998% | 0.0577% | 0.0814% | 0.3952% | \(1.687\times10^{-6}\) | \(1.161\times10^{-5}\) |

All shape, local-\(\phi\), integrated-energy, mass, and far-subtracted local
composition gates pass for all three successive pairs.  The largest-box
pair also passes the registered \(2\times10^{-5}\) absolute-composition
diagnostic.

The isolated-box far-field values approach the requested baseline as the box
grows.  At \(192^3\),

\[
x_{B,\rm far}^\alpha=0.006235218007347056,
\]

compared with the registered input baseline

\[
x_{B,\rm registered}^\alpha=0.006219279767278563.
\]

The quantity

\[
\left(x_{B,\rm far}^\alpha-x_{B,\rm registered}^\alpha\right)L^3
\]

has only 0.6175% relative spread over all four boxes.  This confirms the
expected \(L^{-3}\) uniform inventory offset of a fixed isolated particle.

## Decision

```text
shape_box_size_status=PASS
integrated_elastic_energy_box_size_status=PASS
portable_local_composition_status=PASS
far_offset_inventory_scaling_status=PASS
absolute_raw_composition_status=FAIL_ABSOLUTE_RAW_COMPOSITION_BOX_SIZE_V1
final_status=PASS_ELASTIC_TARGET_PROFILE_PORTABLE_CORRECTION_BOX_SIZE_V1
```

The library's portable initialization content is therefore:

\[
\phi,\qquad
\delta C_{\rm relax}
=(1-h(\phi))(x_B^\alpha-x_{B,\rm far}^\alpha).
\]

An arbitrary target box must reconstruct its own registered matrix baseline
and use the conserved zero mode to close the exact total inventory.  The
absolute isolated-box `xB_alpha.raw.f64` remains authorized only for a
same-grid direct-load fixture.  The portable PASS does not relabel that raw
field as box independent.

## Evidence

- `evidence/workstation_box192_R11p5_v1/elastic_energy_audit.json`,
  SHA-256
  `9b5b645ce0099a2e9b79e95683ceaafb982d97bd9ddad923e7a5fe3f321d214d`;
- `evidence/workstation_box160_192_v2/box_size_audit.json`,
  SHA-256
  `0cdf40ff603aba98866633e34e21b213affd5f7f18f5ca60abf5c143ac727a15`;
- `evidence/workstation_box_series_96_128_160_192_v3/box_series_audit.json`,
  SHA-256
  `02afd6593bc1bb4863e4dec3605122733f1b71a7ef173295dd5770fa7ebf2ec4`;
- `evidence/workstation_box_series_96_128_160_192_v3/final_terminal_output.txt`,
  SHA-256
  `ffaf7d9c37733d66118f32f4a1014e2f12f8c23619ed4596f0e11e8915d0ef7`.
