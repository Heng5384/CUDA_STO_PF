# Production-candidate decision (interim)

This decision is intentionally not final. The V2 fixture and its 6--12 h
preflight are qualified, but the authoritative 12--48 h continuation (job
72915) is still pending GPU resources. The old elastic continuation is not
eligible because it does not use the exact no-elastic qualified path.

`qualified_realistic_effective_fixture=PASS_STATIC_AND_PREFLIGHT`
`qualified_6h_48h_coarsening_trend=PENDING_OFFICIAL_CONTINUATION`
`final_status=PENDING_12H48H_RUN`

The following artifacts are intentionally withheld until the official
same-binary continuation finishes: `population_time_series.csv`,
`particle_trajectories.csv`, `matrix_composition_time_series.csv`,
`mass_and_zero_mode_time_series.csv`, and `t380_6h_48h_trend_analysis.md`.
The exploratory 72831 files are not substituted for these artifacts.
