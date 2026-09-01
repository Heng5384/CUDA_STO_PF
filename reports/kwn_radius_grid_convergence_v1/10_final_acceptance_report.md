# Final acceptance

Top-level status: `DISCRETE_EVENT_SENSITIVITY_GATE_REVIEW_REQUIRED`.

## Independent state closure

- Baseline reproduction: `PASS_RADIUS_GRID_BASELINE_REPRODUCTION`
- Positivity/conservation: `PASS_KWN_POSITIVITY_CONSERVATION_48H`; max uniform-fixture ledger residual `1.55406e-16`, minimum bin `0` m⁻⁴
- Authority requalification: `BLOCKED_RADIUS_GRID_NOT_PASSED`
- Initial projection: `PASS_INITIAL_PSD_PROJECTION_CONSERVATION`; moment quadrature remains diagnostic-only
- Lower-bound closure: `PASS_LOWER_BOUNDARY_TAG_AGGREGATE_AND_LEDGER_CLOSURE`

## Radius-grid decision

Fixture ladder `[100, 200, 400, 800, 1600, 3200]`: `FAIL_KWN_RADIUS_GRID_CONVERGENCE`. Smooth ladder `[100, 200, 400, 800, 1600, 3200]`: `PASS_KWN_RADIUS_GRID_CONVERGENCE`. Final fixture pair `1600_vs_3200`; authority grid `NONE`.

| Primary metric | 48 h relative error | P5 <=2% | Full-time maximum |
| --- | --- | --- | --- |
| N_m0_m3 | 7.684% | FAIL | 10.642% |
| Rmean_m | 2.507% | FAIL | 3.514% |
| Rmean3_m3 | 8.394% | FAIL | 12.046% |
| Sv_m_inv | 2.782% | FAIL | 3.903% |
| f_beta | 0.065% | PASS | 0.122% |
| matrix_xB | 0.314% | PASS | 0.560% |

## Root cause and coupling boundary

Root causes: `IMPLICIT_UPWIND_NUMERICAL_DIFFUSION, DISCRETE_EVENT_SENSITIVITY_IDENTIFIED`. KWN 48 h completion is included in `6/6` uniform fixture runs; restart status: `NOT_RUN_P5_GATED`.

Beta-only comparison: `BLOCKED_RADIUS_GRID_NOT_PASSED`; direction `BLOCKED_RADIUS_GRID_NOT_PASSED`; timescale `NOT_CLAIMED`; mean-field gap `NOT_ASSESSED`.

Frozen CUDA/PF evidence was reused without a CUDA rerun; `PF_SOURCE_MODIFIED=false`; no thermodynamic, D(T), gamma, PSD, or D-scale retuning was performed. `HISTORICAL_AS_RUN_AUTHORITY_UNRECOVERED` and `HISTORICAL_12H_PSD_NOT_RECOVERED` remain in force.

P0 blocker: Exact six-particle fixture P5 is unresolved after the registered final pair; human gate review is required.

Next action: Do not start GP release; preserve the frozen evidence and obtain an explicit human decision on the discrete-event P5 gate.

Local GP release: `LOCAL_GP_RELEASE_NOT_AUTHORIZED`.

Evidence/provenance: `analysis_provenance.json`, `grid_pair_errors.csv`, `grid_psd_distances.csv`, `dissolution_event_times.csv`, and the immutable per-run manifests under `outputs/kwn_radius_grid_convergence_v1/runs/`.
