# Staging vs Canonical Nucleus Library Audit

## Compared artifacts

| Artifact | Rows | Columns | SHA-256 | Status |
|---|---:|---:|---|---|
| `data/nucleus_library/nucleus_library.csv` | 16 | 47 | `c93f8089d936db76afac777b2e4c2917aca36e3ef54c449d401cf8cf21d0ac8d` | committed reference |
| `data/nucleus_library/nucleus_library.json` | 16 | schema v1 | `3570b3b31a337a87982096da489bbf5294a297b3c67e06948a6a7419b38a7f96` | committed mirror |
| `data/nucleus_library/nucleus_library.with_Zr.staging.csv` | 18 | 70 | `b9974b16a1ee5a66febda8c20d053b9b610ca3f966cbbd66c4921a4101310c35` | untracked staging |
| `data/nucleus_library/nucleus_library.with_Zr.staging.json` | 18 | list | `cefaa6126f0172b4da51845a80835e90a4783a9fe35df84aab0d06a98d2fccc0` | untracked staging |

## Result

The files are not equivalent. All 16 shared IDs have at least two changed
barrier fields. `nlib_00006` differs in 36 shared fields, including seed radius,
bridge metadata, geometry, source provenance, and validity flags. The staging
variant also adds 23 fields for Zr/attachment-rate/bridge metadata and adds
`nlib_dc_T400_xB003` plus `nlib_dc_T450_xB003`, which do not exist in canonical.

The five blocked overlays target T380, T400, and T450 with `xB=0.03`; the T400
and T450 dynamic-continue rows are staging-only. Replacing the staging path with
canonical would therefore change available entries, barrier values, seed geometry,
profile provenance, and selector behavior. It is explicitly rejected.

## Scientific status

The canonical library remains `VALIDATION_REFERENCE_ONLY`; its entries are not
production-accepted. The staging library is `BLOCKED_SEMANTICALLY_DIFFERENT_LIBRARY`.
It may become a separately named canonical variant only after a dedicated Zr and
dynamic-continue audit supplies a CSV/JSON mirror, reproducible profiles, checksums,
and acceptance evidence. No staging file was moved, deleted, or committed here.
