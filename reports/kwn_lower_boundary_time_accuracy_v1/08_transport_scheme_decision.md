# Transport-scheme decision

No production transport replacement is selected from an implicit-stability observation.  A donor-bound SSPRK2 reference must be whole-domain positivity-safe; an active-only cap is not presented as such a reference. The actual active-CFL ladder and donor-bound reference record are below.  A frozen boundary-state rate is not a universal feasibility bound and does not by itself establish backward-Euler temporal diffusion.  The next action depends on the actual ladder result: complete its diagnostic budget when incomplete, or evaluate a conservative characteristic or semi-Lagrangian remap only after a genuine source-reported min-dt block.

```json
{
  "actual_implicit_brute_force_policies": [],
  "explicit_reference": {
    "actual_implicit_brute_force_policies": [],
    "implicit_ladder_status": "INCOMPLETE_DIAGNOSTIC_BUDGET",
    "initial_global_face_index": 0,
    "initial_lower_boundary_velocity_m_s": -11767189.594675226,
    "reason": "A whole-domain donor-bound SSPRK2 reference would require dt <= 6.72388328686019683e-20 s from the actual physical-Rmin lower face at the canonical initial state, or at least 2.56994347801493696e+24 accepted steps for 48 h.  It is therefore not implemented or relabelled as an active-only explicit proxy.",
    "reference_solver": "EXPLICIT_UPWIND_DONOR_BOUND_SSPRK2",
    "same_physical_Rmin_contract": true,
    "status": "BRUTE_FORCE_DONOR_BOUND_NOT_PRACTICAL",
    "whole_domain_donor_bound_dt_s": 6.723883286860197e-20,
    "whole_domain_donor_bound_projected_48h_steps": 2.569943478014937e+24,
    "whole_domain_donor_bound_rate_s_inv": 1.4872358090364217e+19
  },
  "implicit_ladder_status": "INCOMPLETE_DIAGNOSTIC_BUDGET"
}
```
