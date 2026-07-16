# Internal reject forensics

The frozen zero-reject reports and their FAIL conclusion are unchanged.
This report only classifies the already-recorded internal trials.

## Counts

- `B_TRANSPORT_NONLINEAR_ITERATION_LIMIT`: 12
- `C_TRANSPORT_LINE_SEARCH_STAGNATION`: 4

## Spatial clustering

- worst cell `244`: 12 rejects
- worst cell `267`: 3 rejects
- worst cell `243`: 1 rejects

## Temporal clusters

- `dt8`: 473, 599, 651, 693, 715, 755, 786, 806, 838, 857, 887, 916, 946, 966-967, 989

## Evidence limits

- The frozen baseline did not hash every rejected rollback. The new default-off contract adds this evidence; the forced rollback smoke is bitwise PASS.
- Per-iteration cold residuals were not stored. The catalog reports accepted nonlinear trial history and the final method-consistent cold audit when reached.
- Cell mobility was not stored per cell; the frozen reject line only recorded its global range. No value is reconstructed or fabricated.
