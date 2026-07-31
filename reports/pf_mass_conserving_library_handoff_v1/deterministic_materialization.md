# Deterministic materialization audit

The formal V1 fixture was frozen from the already qualified exact-library V2
materialization without regenerating any physical field.  Every raw field was
copied byte-for-byte; source and frozen hashes are identical.

| Field | SHA-256 |
|---|---|
| `phi.raw.f64` | `a650c253a0576033371e36fd157ccfb224f3b612cac0fac29f887fe431f469de` |
| `h_phi.raw.f64` | `4ef5da1e13f0c6c415dcb9d45f1991698044a221d77f310bfaaa084982c4177c` |
| `C_B_tot.raw.f64` | `12474fcf608d6fc525bf47ee4f7b8b9353746350e95c3a96182e208142e5f7e9` |
| `xB_alpha.raw.f64` | `578a58d14d73cebd1844453903f6b4adb2ef4d99734306ea39ce709454d31514` |
| `Y.raw.f64` | `5e1bcb24b76a8951782b36df0cfea6a3d747b33a05b291c47d0f0bdbaba0b7fc` |
| `dY_dt_prev.raw.f64` | `468a4a459772da4e498bc9635d3a9c9b12584490e1edcc1d914ee7df6448d8b2` |
| `delta_C_relaxation_total.raw.f64` | `50e8807a3f64cd9e85e6f12fcac735dead9eee9422cbd23dae1e64fc831f6188` |

Static tests independently proved:

- reversing manifest particle order leaves fields and manifest byte-identical;
- repeating the same materialization leaves fields and manifest byte-identical;
- non-finite source fields fail closed;
- malformed hashes and structural inputs fail closed.

Evidence:

```text
source_fixture=/home/zhiheng/tmp/pf_elastic_multi_particle_E2_six_v2p2_20260731c
frozen_fixture=/home/zhiheng/tmp/pf_mass_conserving_library_handoff_fixture_v1_20260731
source_fixture_manifest_sha256=bbac0b9ce0521fb525f76d7bea90da1fc6352cb5bac62cda86a06d27d5e54093
frozen_fixture_manifest_sha256=a87e76405bd1b901b6848b6c39d788ff3fe537ca0acea93f99b64ac67e6ced81
all_raw_source_to_frozen_bytewise_equal=true
deterministic_materialization_status=PASS
```

