# Variable-step active-manifold BDF2 derivation

For current accepted step size `h_n`, previous accepted step size `h_{n-1}`, and
`r=h_n/h_{n-1}`, the variable-step BDF2 derivative is

```text
(a0*u_{n+1} + a1*u_n + a2*u_{n-1}) / h_n
a0 = (1 + 2r)/(1 + r)
a1 = -(1 + r)
a2 = r^2/(1 + r)
```

The transport effective step is `h_eff=h_n/a0`. The second-order phase-context extrapolation is
`phi_context=(1+r)phi_n-r phi_{n-1}` before the existing active-manifold normalization. At
`r=1`, an explicit branch returns the original fixed-step values `(3/2,-2,1/2)`, so the accepted
fixed path is arithmetic-order stable.

The candidate restricts accepted step ratios to `[0.25,2]`; the normal PI growth cap is 1.25.
The embedded low-cost estimator uses the variable-step curvature relative to the accepted
increment for `Ctot`, `phi`, and `h(phi)` volume. Each component is normalized by its preregistered
tolerance, and the maximum component controls acceptance and the next-step proposal. A rejected
error trial uses the existing atomic rollback and does not commit state, BDF2 history, controller
state, event state, counters, or diagnostics.

The implementation, rollback, restart, short-window accuracy, and long-window QoI
comparisons are now complete. The estimator is accurate on accepted trajectories,
but it is not production-qualified: the 8000-macro long run rejected `95.225%`
as many internal trials as accepted macros and spent `43.385%` of wall time on
those rejected trials. It therefore remains default-off diagnostic code.
