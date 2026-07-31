# Elastic dynamic microstructure audit V1

## Decision

`PASS_ELASTIC_OBSERVABLES_BLOCKED_EXACT_ELASTIC_ENERGY`

This report was generated from the newest `audit_pf_dynamic_microstructure_metrics_v1.py` definitions, with the elastic-run adapter. It processes the registered 43 snapshots from 6 h through the exact 48 h endpoint on the cluster.

The particle count, h-volume closure, periodic identity tracking, radius/PSD, Sv, M6, matrix xB/xAg and time derivatives are directly auditable. The elastic VTK run is not downgraded to elasticity-off: exact elastic energy is explicitly **not claimed**, because the VTK files contain phi/xB but not the displacement/stress state.

| gate | result |
|---|---|
| snapshots | 43 registered 6--48 h snapshots |
| particle h-volume closure | PASS (max abs 5.14446e-07) |
| PSD count closure | PASS |
| M6 histogram closure | PASS |
| derivatives | PASS |
| exact elastic energy | BLOCKED: runtime elastic-energy trace not present in VTK |

## Endpoints

| metric | 6 h | 48 h |
|---|---:|---:|
| particle count | 96 | 6 |
| beta volume fraction | 0.023929545 | 0.0237833127 |
| mean radius (nm) | 9.54047 | 23.5174 |
| matrix xB (h<0.005) | 0.00622 | 0.00636834614 |
| matrix xAg | 0.00620071577 | 0.00634813258 |
| Sv (nm^-1) | 0.00742529498 | 0.00287664327 |
| M6 (nm^3) | 5.36260544 | 100.0022 |
| chemical + interface energy (J/m^3) | -191143.292 | -889433.202 |

## Runtime qualification carried with the VTK analysis

The completed elastic continuation reports `PASS_PF_CONSERVED_Y_ZERO_MODE_V1`, exact final mass equality, checkpoint/restart provenance restored and validated, `gp_enabled=false`, and average continuation throughput about 0.422442 s/step. These runtime records are copied into `audit_summary.json`; they are not inferred from VTK.

## Outputs

- `microstructure_time_series.csv`
- `particle_psd_samples.csv`
- `psd_histogram.csv`
- `audit_summary.json`
- `status.txt`

The exact elastic-energy gate remains open until a runtime elastic-energy/stress trace is exported. This is the only intentional blocker in this report.
