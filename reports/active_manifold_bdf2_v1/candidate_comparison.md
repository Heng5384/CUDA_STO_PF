# Active-manifold context candidate comparison

| Candidate | Feasibility | Free-set order | Cost | Decision |
|---|---|---|---|---|
| A tangent-cone velocity | Linearized only; can retain nonlinear residual q defect | Exact away from event | Low | Reject as primary |
| B q-alpha active set + phase endpoint tangent cone | Exact complementarity manifold with one-sided h inverse and exact pure-alpha endpoint | Bitwise original extrapolate away from active branches | Low | **Selected** |
| C constrained fixed-C phase | Feasible but mixes numerical x-min/phase-box policy into transport context and needs a force solve to be complete | Exact away from event | Medium | Retain as oracle only |

Candidate B is not `phi=clamp(phi_raw)`.  It is the semismooth branch
solution of the local complementarity relation `q_E=C_E-h(phi_E)=0`.
Its pure-alpha endpoint branch is the tangent-cone solution `phi_E=0`
only for outward motion at storage-equivalent endpoint states.
It changes no authoritative state and leaves all free cells exactly on
the original second-order formula.

Frozen checkpoint metrics:

- `dt8`: adjusted 4 cells; min q=-1.205e-15; free-set Linf difference=0.000e+00.
- `dt16`: adjusted 0 cells; min q=-9.361e-15; free-set Linf difference=0.000e+00.

`selected_context_candidate=QALPHA_ACTIVE_SET_PREDICTOR`
