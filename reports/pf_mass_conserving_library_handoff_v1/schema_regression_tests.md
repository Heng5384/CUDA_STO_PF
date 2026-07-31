# Schema and negative regression tests

All 18 required gates passed.

| # | Required test | Result |
|---:|---|---|
| 1 | exact registered radius load | PASS |
| 2 | unregistered radius rejection | PASS |
| 3 | library hash mismatch rejection | PASS |
| 4 | duplicate particle ID rejection | PASS |
| 5 | profile support overlap rejection | PASS |
| 6 | periodic-image overlap rejection | PASS |
| 7 | out-of-domain center rejection | PASS |
| 8 | non-finite field rejection | PASS |
| 9 | manifest-order invariance | PASS |
| 10 | repeated bytewise determinism | PASS |
| 11 | per-particle library traceability | PASS |
| 12 | machine-precision global initial inventory | PASS |
| 13 | runtime zero-mode provenance | PASS |
| 14 | continuous/restart raw state byte equality | PASS |
| 15 | fixture/library checkpoint provenance mismatch rejection | PASS |
| 16 | first dynamic step leaves profile free to evolve | PASS |
| 17 | GP/Birth/release/new beta nucleation all OFF | PASS |
| 18 | no common-equilibrium optimizer invoked | PASS |

Additional backward-compatibility checks:

```text
legacy_V2_static_regression=PASS_19_OF_19
legacy_V2_static_regression_artifact_sha256=d42507488e43f2fe8edd71b0d07585d6d7cc61e08d92d31beed7767907909baf
conditional_static_regression=PASS_15_OF_15
conditional_static_regression_artifact_sha256=c6d9dbc5caac6e4463f8a73c88b28e3ff41e7b9707a2bd5725754529217982aa
checkpoint_V3_unit_test=PASS_PF_ZERO_MODE_CHECKPOINT_PROVENANCE_V3
checkpoint_unit_artifact_sha256=9edba4476e98b7648c68a30eee3dbf9f49a058dd30e45aea1c15726b91bef173
```

The old V2 materializer/schema was not reinterpreted or rewritten.  New
policy fields exist only in the new conditional-handoff manifest.

