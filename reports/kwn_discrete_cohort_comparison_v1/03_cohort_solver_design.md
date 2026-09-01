# Discrete-cohort solver design

Each cohort retains its initial ID, frozen PF t=0 equivalent radius, active state, dissolution time, returned inventory and a strictly derived current inventory.  No cohort is binned, projected, split or smoothed.

The physical fixture count and weight are exactly one particle per box.  The frozen global diffuse PF beta bucket and the sum of six sharp component-equivalent spheres are compared only as a strict initial-identity gate; its current status is `FAIL_BETA_ONLY_INITIAL_STATE_IDENTITY`.  No diffuse-volume-derived number-density scale is applied, and radii, D(T), gamma and thermodynamics remain unchanged.

The RHS uses the existing `growth_rate_m_s`, exact validation-contract curvature equilibrium, D(T), molar volumes and lower edge.  Matrix xB is algebraically recovered from total inventory after every RHS evaluation.  A one-sided radius characteristic ends at the Rmin convention without evaluating an invalid sub-Rmin thermodynamic state; the remaining Rmin inventory is returned to the matrix at the accepted event.
