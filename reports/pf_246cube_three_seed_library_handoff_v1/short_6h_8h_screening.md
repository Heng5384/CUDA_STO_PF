# Three-replicate 6--8 h short screening

All three independently placed, hash-pinned fixtures reached registered step
7266 at age 8.000019169100993 h.  No run used a shortened final timestep.

| replicate | N at 6 h | N at 8 h | qualified dissolutions | resolved merges | final beta VF | final mean R (nm) | final Sv (nm^-1) | final M6 (nm^3) | final far-field xAg |
|---|---|---|---|---|---|---|---|---|---|
| A | 96 | 60 | 36 | 0 | 0.0227586238629 | 10.6254419417 | 0.00596732871612 | 9.2129133545 | 0.00738837132637 |
| B | 96 | 54 | 42 | 0 | 0.0228309240198 | 11.1310448394 | 0.00582104678046 | 9.95186793732 | 0.00731452799925 |
| C | 96 | 58 | 37 | 1 | 0.0227691596308 | 10.6359822479 | 0.00586483726352 | 9.69260431821 | 0.00737718442361 |

Relative changes below use the exact materialized step-0 state.  The phi
columns retain the registered initial-reconstruction diagnostic at step 256
and its accumulated value at step 7266.

| replicate | delta beta VF / initial | delta mean R / initial | delta Sv / initial | delta M6 / initial | phi L1 at 256 | phi L1 at 7266 |
|---|---|---|---|---|---|---|
| A | -0.0253642192206 | 0.123401513927 | -0.182731990633 | 0.796748022144 | 0.0350945671851 | 0.705542847255 |
| B | -0.0222684334255 | 0.176857489748 | -0.20276657386 | 0.94086048711 | 0.0358409778077 | 0.779258002276 |
| C | -0.0249129173921 | 0.124515960725 | -0.196768858699 | 0.890300108873 | 0.0343798709959 | 0.720924750982 |

Identity propagation used maximum voxel overlap of periodic connected
components.  A disappearance was accepted only with at least three registered
states, two strictly decreasing final transitions, and final h-volume no more
than half its initial value.  A many-to-one overlap is retained as an explicit
persistent lineage group and must pass the registered multi-threshold
merge-aware audit.  Splits, new components, weak merge edges, and unqualified
dissolutions remain fail-closed.

| replicate | dissolutions | resolved merges | splits | new components | unqualified dissolutions | unqualified merges |
|---|---|---|---|---|---|---|
| A | 36 | 0 | 0 | 0 | 0 | 0 |
| B | 42 | 0 | 0 | 0 | 0 | 0 |
| C | 37 | 1 | 0 | 0 | 0 | 0 |

The stable connected-component labels were joined back to each fixture's
registered `P000`--`P095` identity.  The largest periodic step-0 centroid
discrepancy in A/B/C was
`2.9808940330728761e-13 nm`.

Any original strict-lineage block caused by a many-to-one overlap is preserved
as historical evidence.  Qualification is restored only by the supplemental
explicit-parent, persistent-group, multi-threshold audit; the strict result is
not overwritten or reinterpreted as though it never occurred.

The complete PSD and P000--P095 lineages are retained in the registered CSV
files.  Different random fixtures are not required to follow the same
trajectory.  Far-field concentration and agreement with an experimental 48 h
band are scientific diagnostics, not hard gates in this short qualification.

```text
short_6h_8h_status=PASS
```
