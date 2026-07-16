# dt/16 endpoint energy/work NaN forensics

The step-5483 transport residual, phase KKT, mass, bounds, and finite-state
checks pass.  The first nonfinite audit dependency is the artificial mixed
endpoint state `F(C_np1, phi_n)` used to split the total chain rule into a
transport part and a phase part.  Near `q_alpha=0`, reconstructing that mixed
state produces a negative roundoff-scale matrix inventory and nonfinite
matrix-context quantities at beta-side cells 264-266.  Both chain differences
share this energy, so `transport_chain` and `phase_chain` become NaN together;
the physical-context chemical-potential work also becomes nonfinite, making
`explicit_context_work` NaN.

The actual endpoint states, BDF2 transport context, residual, and final energy
remain finite.  This is an audit-path algebra problem exposed by the same
capacity transition, not a nonfinite accepted PF state.  The stable replacement
must evaluate the combined endpoint chain remainder directly from admissible
endpoint energies and the already finite method work terms.  It must not clamp,
skip, or zero a cell.
