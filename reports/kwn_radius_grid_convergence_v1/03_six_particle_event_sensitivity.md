# Six-particle event sensitivity

The per-class event table uses passive tags that receive the exact production implicit M-matrix update with frozen accepted-step velocities. Lower-bound crossings are reported as accepted-step endpoints, not as fabricated continuous-time events.

Fixture ladder status: `FAIL_KWN_RADIUS_GRID_CONVERGENCE`. The table focuses on final pair `1600_vs_3200`; the raw `class_events.csv` retains every grid/time/class row.

The earliest tagged route is class `R8nm` on the 3200.0-bin grid at accepted endpoint 0.00730797 h. It is an `Rmin` face-flux tail event: it directly changes M0/N, while its fixed-pivot terminal M3 inventory is separately returned to the matrix ledger.

| Grid | Class | R0 / effective R0 (nm) | Projected bins | 48 h occupied bins / sign | First Rmin-face endpoint (h) | Survival at scheduled times | 48 h returned β inventory (mol m⁻³) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1600 | R10.5nm | 10.5/10.5 | 926;927 | 1600 (0–1599) / DISSOLUTION | 1.63235 | 0h:1; 0.1h:1; 0.393177h:1; 1h:1; 3h:1; 6h:1; 12h:0.9761; 24h:0.5595; 48h:0.2632 | 0.0180246 |
| 1600 | R8nm | 8/8 | 844;845 | 273 (573–845) / NEUTRAL | 0.0143366 | 0h:1; 0.1h:1; 0.393177h:1; 1h:0.9837; 3h:6.45e-24; 6h:3.291e-106; 12h:0; 24h:0; 48h:0 | 0.024465 |
| 1600 | R9.5nm | 9.5/9.5 | 896;897 | 1600 (0–1599) / DISSOLUTION | 1.42429 | 0h:1; 0.1h:1; 0.393177h:1; 1h:1; 3h:1; 6h:0.1219; 12h:2.493e-06; 24h:8.415e-09; 48h:5.661e-10 | 0.024465 |
| 3200 | R10.5nm | 10.5/10.5 | 1852;1853 | 3173 (0–3172) / DISSOLUTION | 1.54385 | 0h:1; 0.1h:1; 0.393177h:1; 1h:1; 3h:1; 6h:1; 12h:0.9973; 24h:0.6261; 48h:0.2852 | 0.0174447 |
| 3200 | R8nm | 8/8 | 1690;1691 | none / NEUTRAL | 0.00730797 | 0h:1; 0.1h:1; 0.393177h:1; 1h:0.9985; 3h:8.684e-41; 6h:2.431e-197; 12h:0; 24h:0; 48h:0 | 0.0244036 |
| 3200 | R9.5nm | 9.5/9.5 | 1792;1793 | 3141 (0–3140) / DISSOLUTION | 1.53241 | 0h:1; 0.1h:1; 0.393177h:1; 1h:1; 3h:1; 6h:0.06713; 12h:7.722e-11; 24h:1.723e-15; 48h:2.348e-17 | 0.0244036 |

Flux route: implicit-upwind `Rmin` face (bin 0) → conservative matrix inventory ledger. It directly changes M0/N; the fixed-pivot beta inventory representation is recovered in the matrix ledger without a negative-bin clamp. For PF Case A, component first-absence is checkpoint-bounded and remains fail-closed if an unresolved merge/split event appears.
