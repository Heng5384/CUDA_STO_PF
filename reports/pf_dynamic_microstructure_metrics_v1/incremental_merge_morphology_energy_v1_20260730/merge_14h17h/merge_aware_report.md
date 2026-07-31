# Merge-aware 15--16 h particle audit

`PASS_MERGE_AWARE_GROUP_TRACKING`

The audit uses periodic h(phi)>1e-4 components and writes explicit voxel-overlap parent/child edges. A merge is promoted to a persistent lineage group rather than silently treating a parent as dissolution; a split remains fail-closed.

- snapshots: 14 h, 15 h, 16 h, 17 h
- merge edges: 2
- merge groups: 1
- split edges: 0
- unexplained tracker ages: []
- fail-closed: False

## Merge edges

|parent age|parent id|child age|child id|overlap voxels|parent fraction|child fraction|
|---:|---:|---:|---:|---:|---:|---:|
|15.0|46|16.0|46|43318|1|0.604426|
|15.0|68|16.0|46|24297|0.954433|0.339022|

## Persistent merge groups

|group|parents|child|h-volume before (nm3)|h-volume after (nm3)|change|
|---|---|---:|---:|---:|---:|
|MG_15h_46_68_to_46|46+68|46|36205.8129|38097.2375|0.0522409|
