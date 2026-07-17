# Frozen reject replay

The dt/4 replay uses the retained full nonlinear iteration trace. The 16-row dt/8 catalog originates from `active_manifold_bdf2_v1/workstation_runs/dt8_active_manifold_fixed_qualification_1000`, not the later 4000-step bounded-retry trajectory; its full per-iteration file was not retained. Those rows therefore use the frozen condensed trace, and unavailable endpoint/wall quantities remain explicitly `NOT_AVAILABLE` rather than reconstructed.

| Gate | Outcome | Classification | Count |
|---|---|---|---:|
| G10 | CONVERGED_AT_RELAXED_GATE | RESIDUAL_FLOOR_ONLY | 31 |
| G12 | REJECTED_AS_FROZEN | ITERATION_LIMIT_WITHOUT_GATE_CROSSING | 17 |
| G12 | REJECTED_AS_FROZEN | LINE_SEARCH_DIRECTION_FAILURE | 13 |
| G12 | REJECTED_AS_FROZEN | OTHER | 1 |
| G8 | CONVERGED_AT_RELAXED_GATE | RESIDUAL_FLOOR_ONLY | 31 |
| G9 | CONVERGED_AT_RELAXED_GATE | RESIDUAL_FLOOR_ONLY | 31 |

All relaxed gates are also exercised end-to-end from the byte-identical common checkpoint in the 20-case matrix; those trajectories, rather than condensed replay, are authoritative for endpoint accuracy and wall time.
