# Dynamic-Continue Library Provenance Audit

## Audited entries

| Entry | Temperature | xB | Profile source | Bridge metadata | Status |
|---|---:|---:|---|---|---|
| `nlib_00006` | 380 C | 0.03 | local synthetic runtime profile derived from staging row | metadata + summary present; source history not packaged | `BRIDGE_VALIDATED` reference only |
| `nlib_dc_T400_xB003` | 400 C | 0.03 | local synthetic runtime profile derived from staging row | metadata + summary present; source history not packaged | `BRIDGE_VALIDATED` reference only |
| `nlib_dc_T450_xB003` | 450 C | 0.03 | local synthetic runtime profile derived from staging row | metadata + summary present; source history not packaged | `BRIDGE_VALIDATED` reference only |

## Provenance findings

The staging rows identify dynamic-continue source directories and continuation
labels, but those original source trees are not in the repository. The available
profile directories contain `faceted_family_profiles.csv`, `seed_profile_metadata.json`,
and `source_dyn_dir/summary.txt`; their README files explicitly describe them as
minimal synthetic runtime-compatible profiles rather than VTK-reconstructed faceted
profiles.

The formal bundle therefore preserves the scientific row values and the profile
metadata needed by the current CUDA consumer, while replacing local source paths
with `EXTERNAL_SOURCE_NOT_PACKAGED`. This is sufficient for a reproducible runtime
reference input, not sufficient for a claim that the original continuation chain
has been fully archived.
