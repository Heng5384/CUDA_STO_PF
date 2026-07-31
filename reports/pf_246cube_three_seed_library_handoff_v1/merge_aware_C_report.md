# Merge-aware multi-threshold lineage audit

`PASS_246CUBE_RESOLVED_MERGE_AWARE_LINEAGE_V1`

The registered `h>1e-4` identity contract remains active. Detected many-to-one overlaps are retained as explicit persistent lineage groups; they are not silently counted as dissolution. Splits, new components, weak parent mappings, and unqualified dissolutions remain fail-closed.

| threshold | initial N | final N | dissolutions | merges | splits |
|---:|---:|---:|---:|---:|---:|
| 0.0001 | 96 | 58 | 37 | 1 | 0 |
| 0.001 | 96 | 59 | 37 | 0 | 0 |
| 0.005 | 96 | 59 | 37 | 0 | 0 |

Resolved groups:

| group | threshold | step | age (h) | classification |
|---|---:|---:|---:|---|
| MG_P063_P086 | 0.0001 | 6656 | 7.83211224739 | DIFFUSE_TAIL_NECK_ONLY |

Final V2 analysis provenance is frozen in
`merge_aware_C_analysis_provenance.sha256`.  It records the exact
merge-aware and screening scripts, tracker source and binary, fixture and
component identities, final PF checkpoint, both final audit JSON files, and
the two authoritative V2 PASS status files.  The earlier failed script and
top-level BLOCKED status remain preserved separately as historical evidence.
