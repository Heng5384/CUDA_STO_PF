# `physical_units_config.json` Portability Audit

## Findings

The file is an untracked runtime census with 13 `run_configs`. Each record has a
relative `source_file` under `Results/` and a machine-specific `source_params`
reference. The latter is an absolute local path in every record. The `Results`
files are generated event logs, while the parameter references are provenance-only
links to local run inputs; neither is a canonical physical calibration input.

The file is not consumed by the CUDA executable or by the canonical Unit_Psedobinary
input path found in this audit. Replacing the local references with invented
relative paths would make the historical census look portable without making its
provenance reproducible.

## Decision: `EXCLUDED_EXTERNAL_DEPENDENCY`

The original file remains untouched and untracked. No portable replacement or
example was added because there is no current consumer contract requiring it. A
future template should contain run identifiers and explicit placeholders for
external logs/params, not copied local paths, and must be labeled as provenance
only rather than physical-input acceptance.
