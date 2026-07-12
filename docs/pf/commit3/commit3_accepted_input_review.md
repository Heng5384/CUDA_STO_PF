# Commit 3 Accepted Input Review

## Scope

This commit is limited to the authoritative ten-file candidate list from the
read-only repository audit. It does not alter PF equations, Ctot/qalpha
mathematics, CNT equations, GP ledger semantics, RSMD/S3 behavior, physical
calibration, or seed profiles.

## Candidate decisions

| Candidate group | Count | Decision |
|---|---:|---|
| Canonical library/physical/catalog inputs | 5 | Commit |
| After-quench and physical-CNT parameter overlays | 5 | Block |

The five parameter candidates reference
`data/nucleus_library/nucleus_library.with_Zr.staging.csv` and also use runtime
`Results/` cache paths. That staging file was not in the authoritative
candidate list and is not portable as part of this commit. They remain
untracked and unchanged for a later, separately audited portability fix.

`physical_units_config.json` is excluded. The prior audit found 13 local
user-home paths; it was not modified or staged.

## Accepted scope

The committed five files are small and parseable: the canonical nucleus CSV and
JSON mirror, the compact SI diffusivity table, and root catalog/selection
metadata. The nucleus library is explicitly `VALIDATION_REFERENCE_ONLY`: all
entries currently report `production_valid=false` and
`bridge_validation_status=MISSING`. The root selector metadata records an
explicit fallback, so this commit must not be read as production nucleation
acceptance.

## Validation

- JSON parsing: PASS for catalog, selection, and library mirror.
- CSV parsing: PASS for library and physical diffusivity table.
- CSV/JSON library IDs: 16/16 identical.
- Nucleus library validator: `17` passes, `0` failures.
- Unit conversion smoke: generated schema-v1 params in `/tmp` successfully.
- Candidate privacy scan: no embedded credential material or local paths in the
  five committed files.
- No large file, runtime log, raw CSV, report tree, or staging library is in
  the commit.
