# AQ-to-6 h scattering-equivalent-radius decision

## Answer

At 300.00 K, the frozen Yu AQ two-population calculation gives **1.666662489 W m^-1 K^-1**.  Its exact check at 303.06 K is 1.65602381323 W m^-1 K^-1, matching the existing qualified AQ calculation 1.65602381323 within relative error 7.136e-13.

The Yu AQ small-population-only total is 2.096716695 W m^-1 K^-1 and the big-population-only total is 1.897719484 W m^-1 K^-1.  From the stepwise sequence, adding small objects changes the host-plus-point-defect result by -0.265932262 W m^-1 K^-1; adding big objects after small changes it by -0.430054206 W m^-1 K^-1.  These are sequential Matthiessen differences, not unique additive attribution shares.

Tailoring AQ does not provide the Yu-style unique pair of radius and density inputs.  Its literal T-A/T-B constructions therefore return a conductivity envelope, not a unique Tailoring calculation.  Figure 5c gives a digitized low-temperature estimate of AQ 1.254520 and 6 h 1.291987 W m^-1 K^-1 at 300 K; both are short extrapolations below the plotted 30 degC point and retain a 0.015 W m^-1 K^-1 digitization uncertainty.

For the primary target (Yu AQ at 300 K) in E1, the only scanned crossing is **R = 3.799268608 nm**, on the GEOMETRIC_SIDE side at the median heat-carrying angular frequency.  There are no primary roots in the PF-resolved range (R >= 8 nm), and none in the current 8--11.5 nm library.

Its spherical proxy volume fraction is **0.385921**, well above 0.03.  This violates the stated inventory/object-definition check.  It must not be proposed as a PF seed radius.  The only supported reading is `SCATTERING_EQUIVALENT_RADIUS_NOT_MATERIAL_INVENTORY_EQUIVALENT`.

## Decision

```text
final_status=PASS_SCATTERING_EQUIVALENCE_BUT_INVENTORY_CONFLICT
Yu_AQ_reproduction=PASS
Tailoring_AQ_unique_kappa=false
primary_equivalence_exists=true
primary_resolved_PF_root_count=0
primary_inventory_feasible=False
```

## Recommended next action

Do not convert this effective scattering root into the 6 h PF library.  If the project later seeks to model the low-temperature AQ-to-6 h thermal-conductivity contrast, introduce a separately identified unresolved Ag-rich scattering population (and its independent inventory/observation contract), then compare it against resolved-beta, strain, and any experimentally supplied dislocation background without retuning the frozen Yu host.
