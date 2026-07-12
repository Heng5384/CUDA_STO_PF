# Dynamic-Continue Library v1 Validation

The formal variant is `data/runtime_profiles/dynamic_continue_v1/` and contains
three entries required by the current T380/T400/T450 overlays. It is a projection
of the audited staging rows, not a replacement of the 16-row canonical library.

Validation results:

- entry IDs: 3/3 present;
- T/xB/radius/barrier/geometry fields: copied from staging rows without numerical edits;
- CSV/JSON mirror: exact field/value match for all emitted rows;
- bundle-specific paths: relative and inside the bundle root;
- source/user-home paths: removed from emitted runtime inputs;
- `production_valid` and `bridge_validation_status`: retained from source rows;
- scientific status: `VALIDATION_REFERENCE_ONLY`.

The original dynamic-continue source history is not packaged. A future scientific
archive may add it, but this variant is sufficient only for the current runtime
profile consumer and reference-level portability tests.
