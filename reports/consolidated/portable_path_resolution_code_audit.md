# Portable Path Resolution Code Audit

## Findings

The Python selector has a repository-root resolver in `nucleus_selector.py:15-49`.
It resolves relative selector/catalog paths against the source tree and maps one
legacy absolute prefix. This resolver is not used by the CUDA parameter loader.

The CUDA path is process-CWD based. `main_cuda.cu:21571-21794` copies path-valued
parameters as raw strings; `main_cuda.cu:5353-5355` and `main_cuda.cu:5587-5591`
pass those strings directly to `std::ifstream`. The dynamic-continue cache root is
then concatenated with entry IDs at `main_cuda.cu:5662-5668`. No repository-root,
parameter-file-directory, or explicit input-root normalization occurs.

`nucleus_orchestrator.py:17-19` correctly anchors its own workflow paths to the
script directory, but its runtime paths are a separate Python workflow contract.
It does not repair the C++ runtime contract.

## Decision

No second resolver was added in this portability commit. Adding one would change
the CUDA configuration/runtime boundary and would require a dedicated PF/GP runtime
acceptance test. The blocked overlays remain unmodified.

## Required future contract

1. Resolve authoritative repository inputs from an explicit repository root or a
   parameter-file-aware resolver.
2. Resolve optional inputs from an explicit `input_root`/CLI contract.
3. Keep runtime output and regenerable cache under an explicit output root; never
   infer them from an existing `Results/` tree.
4. Missing authoritative inputs must fail with the resolved path and parameter key.
5. Validate the same parameter from repository root, build directory, and an
   unrelated temporary cwd before enabling any overlay.
