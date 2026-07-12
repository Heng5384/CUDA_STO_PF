# CUDA Runtime Input Path Audit

## Scope

This audit covers the path-valued PF/CUDA runtime configuration used by the five
blocked overlays. The complete key map is in
`cuda_runtime_input_path_map.csv`.

## Current consumers

- `main_cuda.cu:22200-22213` stores runtime library/cache paths as raw strings.
- `main_cuda.cu:5353` opens the barrier CSV directly.
- `main_cuda.cu:5587` opens the runtime nucleus CSV directly.
- `main_cuda.cu:5662-5668` constructs profile, source-dynamic, and metadata paths
  from the cache root and entry ID.
- `main_cuda.cu:5833-5836` rejects a candidate if the profile CSV, metadata JSON,
  or source-dynamic directory is missing.
- `main_cuda.cu:25003-25110` creates runtime output directories; these outputs are
  not valid input fallbacks.
- `main_cuda.cu:27690-27714` reads conservative storage restart raw data when the
  explicit restart field is enabled.

## Implemented contract

`resolve_portable_runtime_input_paths()` now resolves declared roots and `repo:`,
`runtime:`, `bundle:`, `restart:`, `param:`, `output:`, and explicit `external:`
tokens. Runtime-library overlays must use `bundle:` paths when the versioned
bundle selector is enabled. Bundle paths are normalized, kept within the bundle
root, and checked against `bundle_manifest.csv` for version, existence, size, and
SHA-256 before runtime selection.

## Important boundary

The formal bundle is a `VALIDATION_REFERENCE_ONLY` input bundle. It packages the
three available synthetic runtime-compatible profiles; it does not promote GP
growth/release to production acceptance and does not alter any physical values.
