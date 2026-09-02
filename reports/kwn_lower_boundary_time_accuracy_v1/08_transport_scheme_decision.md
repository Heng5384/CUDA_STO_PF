# Transport-scheme decision

No production transport replacement is selected from an implicit-stability observation.  A donor-bound SSPRK2 reference must be whole-domain positivity-safe; an active-only cap is not presented as such a reference. The active physical-Rmin feasibility record and donor-bound reference record are below.  They block the current implicit upwind operator from a 2% cohort qualification; they do not themselves prove a measured backward-Euler temporal-diffusion error.  The next allowed evaluation is a conservative characteristic or semi-Lagrangian remap with the same shared boundary contract.

```json
{
  "explicit_reference": {
    "implicit_ladder_status": "BRUTE_FORCE_CFL_NOT_PRACTICAL",
    "initial_global_face_index": 0,
    "initial_lower_boundary_velocity_m_s": -11767189.594675226,
    "observed_active_CFL_4_dt_s": 2.6892616479016512e-19,
    "observed_active_boundary_rate_s_inv": 1.4873971088388063e+19,
    "reason": "A whole-domain donor-bound SSPRK2 reference would require dt <= 6.72388328686019683e-20 s from the actual physical-Rmin lower face at the canonical initial state, or at least 2.56994347801493696e+24 accepted steps for 48 h.  It is therefore not implemented or relabelled as an active-only explicit proxy.",
    "reference_solver": "EXPLICIT_UPWIND_DONOR_BOUND_SSPRK2",
    "same_physical_Rmin_contract": true,
    "status": "BRUTE_FORCE_DONOR_BOUND_NOT_PRACTICAL",
    "whole_domain_donor_bound_dt_s": 6.723883286860197e-20,
    "whole_domain_donor_bound_projected_48h_steps": 2.569943478014937e+24,
    "whole_domain_donor_bound_rate_s_inv": 1.4872358090364217e+19
  },
  "implicit_boundary_feasibility": {
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
}
```
