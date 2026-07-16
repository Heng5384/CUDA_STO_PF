# Stable endpoint energy evaluation contract

The old split introduces the intermediate `F(C_np1,phi_n)`.  Algebraically,
the only quantity used in the final balance is

`R_chain + W_context = Delta F_nonelastic - <mu_E,Delta C> - <g_np1,Delta phi>`.

The stable evaluator computes this combined expression directly from the two
admissible accepted/trial endpoint energies and the same finite `mu_E` and
phase-force work arrays used by the residual audit.  It does not reconstruct an
off-method mixed endpoint.  Away from active-set events it agrees algebraically
with the old expanded expression:

`[F(Cnp1,phin)-F(Cn,phin)-<mu_n,dC>]`

`+[F(Cnp1,phinp1)-F(Cnp1,phin)-<g,dphi>]`

`+[<mu_n,dC>-<mu_E,dC>]`

`=F(Cnp1,phinp1)-F(Cn,phin)-<mu_E,dC>-<g,dphi>`.

The host unit oracle in `tests/test_bdf2_event_utils.cpp` verifies this
identity to roundoff.  The runtime implementation is selected only with the
default-off event preflight and reports contract
`BDF2_STABLE_ADMISSIBLE_ENDPOINT_WORK_V2`.  At the event it remains finite by
avoiding the undefined mixed intermediate; it does not use `isfinite ? 0`,
skip a cell, clip a field, or relax the energy gate.

## Runtime acceptance

The dt/16 replay evaluates 438 BDF2 endpoint-work rows after the original
failure and all are finite with `pass=1`; it then completes 502/502 accepted
macros. The dt/8 replay also keeps every evaluated BDF2 V2 row finite. Thus the
old dt/16 NaN is closed as an audit-algebra defect. The separate dt/8 event
frequency failure does not originate in this energy expression.
