# Blocked Parameter Dependency Audit

## Scope

Audited the five untracked reference overlays listed in the portability task. The
tracked Commit 3 baseline was verified at `b190ddb`; its ten files are limited to
the audited canonical inputs, manifest, and compact reports. Tracked worktree and
index are clean; unrelated untracked material remains intentionally untouched.

## Common dependency graph

Each overlay enables `use_nucleus_library_selector=1`,
`enable_gp_runtime_library_nucleation=1`, `enable_runtime_nucleus_library=1`, and
`enable_dynamic_continue_bridge=1`. Each points all three library keys to the
untracked `with_Zr` staging CSV and both cache keys to the untracked
`Results/runtime_profile_cache/dx1p0` tree. The graph is tabulated in
`blocked_parameter_dependency_graph.csv`.

## Runtime behavior

`main_cuda.cu:1794-1839` validates that the GP library and dynamic-continue paths
are nonempty when the optional runtime feature is enabled. The barrier CSV is
opened at `main_cuda.cu:5353-5355`; the nucleus CSV is opened at
`main_cuda.cu:5587-5591`. Profile paths are synthesized from the cache root at
`main_cuda.cu:5662-5668`. Thus the cache is a runtime dependency for these
overlays, not a harmless historical annotation.

## Decision

No overlay was modified. A relative path alone would not make the overlay portable,
and changing the library to canonical would change scientific/runtime semantics.
The five overlays remain blocked until a separately audited Zr/dynamic-continue
input bundle and a C++ path contract are accepted.
