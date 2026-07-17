# Adaptive scalar preconditioner validation

The rule uses only an existing mobility reduction and clips `a_ref_eff` and
`D_ref_eff` to the preregistered tenfold interval. Unit tests cover finite and
fallback paths. Runtime V4 completes the 2000-step dt/4 diagnostic 20.5% faster
than V0 but requires 15 internal rejects and 10 fallback macros. V5 reduces
that short-window count to 2/2, yet fails the long-window zero-fallback gate.
Status: `RUNTIME_BENEFIT_OBSERVED_NOT_PRODUCTION_QUALIFIED`.
