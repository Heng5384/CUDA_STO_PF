# q_alpha=0 active-manifold derivation

For `v_B=1`, the local storage map is

`Ctot = h(phi) + q_alpha`, with `q_alpha=(1-h)xB_alpha`.

At the lower branch, the admissible set and complementarity conditions are

`q_alpha >= 0`, `lambda_q >= 0`, `q_alpha*lambda_q=0`.

At fixed BDF2 anchor `C_E=(4 C_n-C_nm1)/3`,

`dq/dphi = -h'(phi)` and the outward normal test is
`delta_C-h'(phi) delta_phi < 0`.

The raw context `phi_E=2 phi_n-phi_nm1` repeatedly has a negative normal
component on the moving beta-side interface.  This is the same moving
lower-capacity manifold found by the event-safe audit; it is not stale history
or an endpoint-energy failure.  The dt/8 log records 240 preflight events over
502 macro steps and the first affected cell migrates with the interface.

The selected context solves the active storage relation exactly:

`q_E=0`, `h(phi_E^M)=C_E`, `phi_E^M=h^{-1}(C_E)`.

Only the non-authoritative transport coefficient context is reconstructed.
`C_n`, `C_nm1`, `phi_n`, and `phi_nm1` remain unchanged.  In free cells the
context is bitwise the original second-order extrapolate.  A one-sided inverse
returns the representably feasible side of the manifold rather than clipping a
physical state.

The production replay also exposed a separate endpoint tangent cone in the
pure-alpha far field. There `phi>=0`, `h(0)=h'(0)=0`, and accepted positive
tails with `h(phi_n)` below the 64-ULP storage resolution are storage-equivalent
to the exact endpoint. If `2*phi_n-phi_nm1<0` points outward, the coefficient
context is `phi_E^M=0`; inward motion remains the original extrapolate. This is
an explicit complementarity branch, not a post-update clamp. Adding it removes
all phase-endpoint preflight fallbacks in both 502-macro replays.

Frozen-state all-cell classes are:

- `dt8`: ENTERING_QALPHA_LOWER=1, FREE=491, PERSISTENT_QALPHA_LOWER=8, UPPER_CAPACITY_ACTIVE=12
- `dt16`: FREE=496, PERSISTENT_QALPHA_LOWER=4, UPPER_CAPACITY_ACTIVE=12

`active_manifold_root_cause_confirmed=true`
