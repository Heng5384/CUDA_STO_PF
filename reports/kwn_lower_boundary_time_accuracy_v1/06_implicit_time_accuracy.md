# Implicit time-accuracy ladder

Status: `BRUTE_FORCE_CFL_NOT_PRACTICAL`.  The reference limit is `No qualified finer implicit trajectory exists after the physical-Rmin active-CFL bound; no cohort reference or 2% pass is claimed.`.  Candidate cheapest cap: `None`.  Measurable current-vs-finest implicit difference: `False`.  Brute-force policies: `['implicit_active_CFL_lte_4']`.

Observed physical-boundary feasibility (where present):

```json
{
  "active_rate_s_inv": 1.4873971088388063e+19,
  "below_configured_min_dt": true,
  "boundary_active": true,
  "configured_min_dt_s": 1e-12,
  "feasibility_kind": "observed_physical_Rmin_active_CFL_bound",
  "observed_boundary_state": {
    "boundary_rate_s_inv": 1.4873971088388063e+19,
    "boundary_velocity_m_s": -11768465.81822008,
    "source": "terminal_state_audit_dt_1_s",
    "step": 249,
    "time_s": 360.0
  },
  "observed_from_policy": "current_implicit_policy",
  "projected_48h_steps_at_current_required_dt": 6.425555510183643e+23,
  "required_dt_s_at_requested_active_cfl": 2.6892616479016512e-19,
  "source_owned_active_CFL_fixed_point_probe": {
    "error": "SolverStateError: CFL-limited step 2.689e-19 s is below configured min_dt_s=1.000e-12 s",
    "runtime_s": 0.022086292000004448,
    "solver_reported_cfl_limited_dt_s": 2.689e-19,
    "status": "REJECTED_BELOW_MIN_DT"
  }
}
```

| policy | requested_active_cfl | evaluation_mode | status | accepted_steps | runtime_s | estimated_48h_steps | estimated_48h_runtime_s | brute_force_active_rate_s_inv | brute_force_required_dt_s | brute_force_projected_48h_steps | brute_force_boundary_active |
|---|---|---|---|---|---|---|---|---|---|---|---|
| current_implicit_policy | UNBOUNDED | accepted_short_trajectory | PASS_EULERIAN_SHORT_RUN | 249 | 4.806518707999999 | NOT_PROJECTED_FROM_UNQUALIFIED_SHORT_RUN | NOT_PROJECTED_FROM_UNQUALIFIED_SHORT_RUN | None | None | None | True |
| implicit_active_CFL_lte_4 | 4.0 | observed_physical_Rmin_active_CFL_bound | BRUTE_FORCE_CFL_NOT_PRACTICAL | 0 | 0.0 | NOT_PROJECTED_FROM_UNQUALIFIED_SHORT_RUN | NOT_PROJECTED_FROM_UNQUALIFIED_SHORT_RUN | 1.4873971088388063e+19 | 2.6892616479016512e-19 | 6.425555510183643e+23 | True |
| implicit_active_CFL_lte_2 | 2.0 | observed_physical_Rmin_active_CFL_bound | BLOCKED_BY_BRUTE_FORCE_CFL_NOT_PRACTICAL | 0 | 0.0 | NOT_PROJECTED_FROM_UNQUALIFIED_SHORT_RUN | NOT_PROJECTED_FROM_UNQUALIFIED_SHORT_RUN | 1.4873971088388063e+19 | 1.3446308239508256e-19 | 1.2851111020367286e+24 | True |
| implicit_active_CFL_lte_1 | 1.0 | observed_physical_Rmin_active_CFL_bound | BLOCKED_BY_BRUTE_FORCE_CFL_NOT_PRACTICAL | 0 | 0.0 | NOT_PROJECTED_FROM_UNQUALIFIED_SHORT_RUN | NOT_PROJECTED_FROM_UNQUALIFIED_SHORT_RUN | 1.4873971088388063e+19 | 6.723154119754128e-20 | 2.570222204073457e+24 | True |
| implicit_active_CFL_lte_0.5 | 0.5 | observed_physical_Rmin_active_CFL_bound | BLOCKED_BY_BRUTE_FORCE_CFL_NOT_PRACTICAL | 0 | 0.0 | NOT_PROJECTED_FROM_UNQUALIFIED_SHORT_RUN | NOT_PROJECTED_FROM_UNQUALIFIED_SHORT_RUN | 1.4873971088388063e+19 | 3.361577059877064e-20 | 5.140444408146914e+24 | True |
| implicit_active_CFL_lte_0.25 | 0.25 | observed_physical_Rmin_active_CFL_bound | BLOCKED_BY_BRUTE_FORCE_CFL_NOT_PRACTICAL | 0 | 0.0 | NOT_PROJECTED_FROM_UNQUALIFIED_SHORT_RUN | NOT_PROJECTED_FROM_UNQUALIFIED_SHORT_RUN | 1.4873971088388063e+19 | 1.680788529938532e-20 | 1.0280888816293829e+25 | True |
| implicit_active_CFL_lte_0.125 | 0.125 | observed_physical_Rmin_active_CFL_bound | BLOCKED_BY_BRUTE_FORCE_CFL_NOT_PRACTICAL | 0 | 0.0 | NOT_PROJECTED_FROM_UNQUALIFIED_SHORT_RUN | NOT_PROJECTED_FROM_UNQUALIFIED_SHORT_RUN | 1.4873971088388063e+19 | 8.40394264969266e-21 | 2.0561777632587658e+25 | True |
