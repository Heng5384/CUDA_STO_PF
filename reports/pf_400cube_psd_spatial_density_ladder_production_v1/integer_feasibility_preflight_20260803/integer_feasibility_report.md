# 400-cube PSD/spatial/density ladder: integer h-volume feasibility audit

This is a read-only pre-materialization audit.  No 400³ fixture or Slurm production job was created.

## Result

`BLOCKED_PROFILE_LIBRARY_RANGE`

The requested target cannot be met by any integer histogram of the frozen 8.00--11.50 nm, 0.25 nm library under the requested relative h-volume tolerance of `1e-10`.

## Why this is a hard gate

For a registered radius R, q=4R is an integer and the ideal equivalent h-volume is `(pi/48) q^3`.  Thus every allowed integer profile histogram lies on that ideal volume lattice, plus the recorded (positive) profile-level corrections.  The nearest ideal lattice point is farther from the requested target than all possible corrections can bridge.

| N | requested absolute tolerance (nm³) | rigorous lower bound on error (nm³) | lower-bound relative error | result |
|---:|---:|---:|---:|---|
| 256 | 0.00015314909 | 0.0231318125 | 1.51041135e-08 | BLOCKED_INTEGER_LATTICE_MISMATCH |
| 512 | 0.00015314909 | 0.0230807208 | 1.50707528e-08 | BLOCKED_INTEGER_LATTICE_MISMATCH |
| 640 | 0.00015314909 | 0.023055175 | 1.50540724e-08 | BLOCKED_INTEGER_LATTICE_MISMATCH |
| 704 | 0.00015314909 | 0.0230424021 | 1.50457322e-08 | BLOCKED_INTEGER_LATTICE_MISMATCH |

The bound is already greater than the tolerance before applying the additional P0/P1/P2/P3 shape constraints.  Changing the PSD optimizer, spatial seeds, profile assignment, or queue cannot remove it.  Profile scaling/interpolation or changing the frozen target would remove the proof's assumptions and is forbidden by the task contract.

## Consequence

All 21 fixtures are blocked at their common static h-volume gate.  The required dual queue submission was deliberately not performed; submitting it would create runs that cannot satisfy the stated fixture contract.  The current V5 pilot was not read from or written to by this audit.
