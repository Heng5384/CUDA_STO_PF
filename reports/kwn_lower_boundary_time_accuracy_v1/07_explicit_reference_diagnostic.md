# Explicit reference diagnostic

Status: `BRUTE_FORCE_DONOR_BOUND_NOT_PRACTICAL`.  A whole-domain donor-bound SSPRK2 reference would require dt <= 6.72388328686019683e-20 s from the actual physical-Rmin lower face at the canonical initial state, or at least 2.56994347801493696e+24 accepted steps for 48 h.  It is therefore not implemented or relabelled as an active-only explicit proxy.

```json
{
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
}
```
