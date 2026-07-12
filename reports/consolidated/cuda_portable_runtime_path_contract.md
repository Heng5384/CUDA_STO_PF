# CUDA Portable Runtime Path Contract

## Roots

| Root | Meaning | Default |
|---|---|---|
| `repository_root` | repository-relative authoritative inputs | inferred from the parameter file ancestry or `--repository-root` |
| `parameter_file_dir` | directory containing the loaded `.params` file | derived, logged only |
| `runtime_input_root` | explicitly declared non-repository runtime inputs | `repository_root` |
| `dynamic_continue_bundle_root` | versioned dynamic-continue input bundle | `repository_root/data/runtime_profiles/dynamic_continue_v1` |
| `restart_root` | declared restart/checkpoint input root | `repository_root` |
| `output_root` | runtime output destination | explicit root, then `CUDA_STO_RESULTS_ROOT`, then legacy `Results` |

## Path tokens

Path-valued runtime parameters use one explicit base: `repo:`, `runtime:`,
`bundle:`, `restart:`, `param:`, `output:`, or `external:`. Unqualified absolute
paths are rejected by the portable runtime resolver. Bundle paths cannot escape
`dynamic_continue_bundle_root`, including through an existing symlink.

## Runtime rules

1. Authoritative library and profile inputs are resolved before runtime selection.
2. A versioned bundle requires `dynamic_continue_bundle_version` and a manifest.
3. Missing, corrupt, wrong-version, wrong-size, or wrong-checksum bundle files are
   fatal; no Results-tree search or alternate nucleus fallback is permitted.
4. Runtime output is created under `output_root` and is never reused as input.
5. Restart inputs remain explicit; missing restart data is a hard read failure.
6. The same `.params` plus the same explicit roots resolves identically from any
   process working directory.
