# Commit 3 Acceptance Report

Commit 3 is intentionally a partial accepted-input commit. Five of the ten
authoritative candidates are staged; five parameter overlays are excluded
because they depend on an excluded staging nucleus library and runtime cache
paths. No substitute files were added.

The committed set contains only compact, parseable inputs and metadata. One
diffusivity table is `ACCEPTED_PHYSICAL_INPUT`; the nucleus library, catalog,
and selected metadata are `VALIDATION_REFERENCE_ONLY`. Their inclusion does
not promote any PF composition mode, GP path, CNT path, or seed profile to
production acceptance.

Recovery commit `7097a74` was verified before this commit. Its scope matches
the repository recovery manifest and its recorded build/restart tests are all
PASS. No recovery files are duplicated here.

The five blocked parameter overlays require a later portability task: either a
new audited canonical library input and repository-relative cache contract, or
explicitly documented external runtime dependencies. `physical_units_config.json`
remains excluded due to local absolute paths.
