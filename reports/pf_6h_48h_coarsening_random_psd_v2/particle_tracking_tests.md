# Particle tracking tests

The tracking contract is maximum voxel overlap, with periodic centroid
distance as the tie-breaker. The 6--12 h audit uses the frozen `h` threshold
`1e-4` on the `246^3` domain.

| experimental age | components | merge/split |
|---:|---:|:---|
| 6 h | 96 | none |
| 7 h | 96 | none |
| 8 h | 70 | none |
| 9 h | 56 | none |
| 10 h | 48 | none |
| 11 h | 42 | none |
| 12 h | 34 | none |

The loss events in this preflight are therefore component disappearance under
the threshold, not an observed connected-component merge or split. The
12--48 h continuation must be analyzed with the same contract before the
final population trajectory is accepted.
