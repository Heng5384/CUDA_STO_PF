# Library mapping audit

## Frozen identities

```text
library_manifest_sha256=58803a8bc6679b823e45e7a7b85df16ae68efa55338d52d4c4151b414a5ef0fe
selection_provenance_sha256=56c44d8f72b27bb462dffe89b59cb2fcb2ff0bf8807dec9d9cac31d2bd7fcbe3
profile_source_tree_sha256=f7855699addf98f9d5aed03af876d851d524fb62c98f5a48df78d1fe561a2a75
profile_binary_sha256=55cf917df94fcf01373d62f54f9ab99715975863ad5dcfb95baa8a517460cda1
conditional_fixture_manifest_sha256=a87e76405bd1b901b6848b6c39d788ff3fe537ca0acea93f99b64ac67e6ced81
```

## Six-particle mapping

| Particle | Entry | Radius (nm) | Center | Profile manifest SHA-256 |
|---|---:|---:|---|---|
| MP001 | R10p5 | 10.5 | (22,5,22) | `ac559f2ff5aa9c32a194b454c9f3a1a41a3732dfc9368ee5cb526bd27c2e625e` |
| MP002 | R10p5 | 10.5 | (54,37,32) | `ac559f2ff5aa9c32a194b454c9f3a1a41a3732dfc9368ee5cb526bd27c2e625e` |
| MP003 | R9p5 | 9.5 | (86,81,70) | `e8909cf192b0250a2a3f81cd8f0c43c724334289d73e82668eb37b888bfad07f` |
| MP004 | R9p5 | 9.5 | (33,81,70) | `e8909cf192b0250a2a3f81cd8f0c43c724334289d73e82668eb37b888bfad07f` |
| MP005 | R8p0 | 8.0 | (95,43,5) | `0c323647c09326d1c39e22d853f6cbda783dbdab72da9e308792cb3bd9b898b2` |
| MP006 | R8p0 | 8.0 | (18,34,56) | `0c323647c09326d1c39e22d853f6cbda783dbdab72da9e308792cb3bd9b898b2` |

All six use `variant_100_identity`.  The mapping also records each profile's
raw `phi` hash, `delta_C_relaxation` hash, isolated effective h-volume, and
canonical assembled-particle inventory.

The periodic hard-core contract is

```text
distance(i,j) > Ri + Rj + 4*lambda_sm
```

and the materialized field contains exactly six connected beta components at
the frozen `h>1e-4` threshold.  Mapping from particle IDs to component labels
is bijective.

## Decision

```text
profile_library_mapping_status=PASS
particle_count=6
unregistered_radius_used=false
interpolation_used=false
scaling_used=false
rotation_used=false
analytic_tanh_used=false
```

