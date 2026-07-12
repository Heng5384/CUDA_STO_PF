# CUDA Path Resolver Test Report

The artifact-level validator passed the bundle mirror, checksum, missing-file,
wrong-version, checksum-mismatch, and traversal tests. The C++ resolver implements
the same hard-failure contract and is ready for workstation compilation.

The valid CUDA startup cases D1-D3 require a compiled `main_cuda` and a complete
base PF parameter file with one of the overlays appended. They were not claimed as
runtime PASS on this macOS host because `nvcc` is unavailable here. The test matrix
records the required workstation executions rather than fabricating their output.
