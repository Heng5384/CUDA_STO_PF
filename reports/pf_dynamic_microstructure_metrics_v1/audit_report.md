# Dynamic microstructure metrics audit V1

## Decision

`PASS_EXISTING_72831_DYNAMIC_MICROSTRUCTURE_AUDIT_V1`

The audit uses the existing 72831 no-elastic 6--48 h trajectory and 43 pairs of
phi/xB VTK snapshots.  No solver output was modified and no new simulation was
run.  The requested 3-lambda and 4-lambda xAg shell values are explicitly
excluded.

## Registered observables

| observable | implementation | gate |
|---|---|---|
| beta volume fraction | mean h(phi), cross-checked against particle h-volumes | PASS |
| particle count | canonical periodic trajectory components | PASS |
| average radius | number mean of h-volume equivalent radii | PASS |
| PSD | per-particle samples plus fixed 2 nm histogram | PASS |
| Sv | sum(4 pi R^2)/V_box | PASS |
| M6 | sum(R^6)/V_box | PASS |
| growing/shrinking | +/- 1% adjacent-snapshot classification; dissolved included in shrinking total | PASS |
| chemical energy | solver-identical bulk excess formula, xB_ref=0.00622 | PASS |
| interface energy | solver-identical centered gradient + double well | PASS |
| elastic energy | exact zero because 72831 elasticity is OFF | PASS_EXACT_ZERO_ELASTICITY_OFF |
| df_beta/dt | nonuniform physical-time finite difference | PASS |
| dxAg/dt | nonuniform physical-time finite difference of matrix xAg | PASS |

## Existing-result endpoints

| metric | 6 h | 48 h |
|---|---:|---:|
| particle count | 96 | 8 |
| beta volume fraction | 0.023929545 | 0.0248527419 |
| mean equivalent radius (nm) | 9.54047 | 21.2647 |
| Sv (nm^-1) | 0.00742529498 | 0.00321798965 |
| M6 (nm^3) | 5.36260544 | 82.8110232 |
| matrix xAg | 0.00620071577 | 0.00526465118 |
| chemical excess energy (J/m^3) | -1423388.79 | -1432284.92 |
| interface energy (J/m^3) | 857337.726 | 529834.653 |
| elastic energy (J/m^3) | 0 | 0 |

The cumulative 6--48 h classification has
7 growing survivors, 1 shrinking survivors,
0 stable survivors and 88 dissolved
resolved particles.  The registered growing/shrinking ratio, with dissolved
particles included in the shrinking total, is 0.0786516854.

## Files

- `microstructure_time_series.csv`
- `particle_psd_samples.csv`
- `psd_histogram.csv`
- `audit_summary.json`
- `status.txt`

## Provenance boundary

This validates the metrics implementation against the already accepted 72831
trend evidence.  The retained 72831 executable hash is
`e5b6adcb489c821529e1a606b9bcff624e6b30565f72c07bce1ba8c30c23c1a3` and its parameter hash is
`e00a17194fd08c8a3d8db286272667c009a9ef908a951c2a9eb1e820f6236178`.  This report does not erase
the previously recorded binary/source mismatch.
