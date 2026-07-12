# Dynamic-Continue Runtime Input Bundle v1

This is a small, versioned CUDA runtime input bundle containing the three audited
reference entries required by the T380/T400/T450 overlays. It is not a production
GP-growth acceptance bundle. Scientific status: `VALIDATION_REFERENCE_ONLY`.

- bundle version: `dynamic_continue_v1`
- source code commit: `1fd1c542746555797754dfcdc2411a4f795c4e93`
- staging library source SHA-256: `b9974b16a1ee5a66febda8c20d053b9b610ca3f966cbbd66c4921a4101310c35`
- profile representation: synthetic runtime-compatible faceted-family profiles
- integrity manifest: `bundle_manifest.csv`

The bundle is addressed through `bundle:<relative-path>` parameter values. The
runtime must verify the manifest before selecting an entry. Historical output-tree
paths and machine-local provenance paths are intentionally absent.
