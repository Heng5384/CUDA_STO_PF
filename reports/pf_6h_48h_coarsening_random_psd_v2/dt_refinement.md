# Timestep refinement

The official preflight includes `dt_code=0.02` and a `dt/2` endpoint run. The
`dt=0.02` final volume diagnostic is `0.0244209151`; the refined run reports
`0.0245743267` at the same physical endpoint. Both retain roundoff-scale mass
residuals. This is recorded as a refinement comparison, not as evidence that
the full 12--48 h trend is already qualified.

`dt_refinement_status=PASS_PREFLIGHT_COMPARISON_PENDING_FULL_TREND`
