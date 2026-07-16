# Internal reject forensics

The frozen zero-reject reports and their FAIL conclusion are unchanged.
This report only classifies the already-recorded internal trials.

## Counts

- `B_TRANSPORT_NONLINEAR_ITERATION_LIMIT`: 5
- `C_TRANSPORT_LINE_SEARCH_STAGNATION`: 9
- `J_OTHER_PROVEN_CAUSE`: 1

## Spatial clustering

- worst cell `244`: 3 rejects
- worst cell `266`: 3 rejects
- worst cell `245`: 3 rejects
- worst cell `267`: 3 rejects
- worst cell `279`: 1 rejects
- worst cell `268`: 1 rejects
- worst cell `278`: 1 rejects

## Temporal clusters

- `dt4`: 1283, 1412, 1536, 1551, 1576, 1596-1597, 1637, 1687, 1950-1954

## Evidence limits

- The frozen baseline did not hash every rejected rollback. The new default-off contract adds this evidence; the forced rollback smoke is bitwise PASS.
- Per-iteration cold residuals were not stored. The catalog reports accepted nonlinear trial history and the final method-consistent cold audit when reached.
- Cell mobility was not stored per cell; the frozen reject line only recorded its global range. No value is reconstructed or fabricated.
