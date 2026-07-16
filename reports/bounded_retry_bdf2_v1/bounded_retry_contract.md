# Bounded-retry contract

`ACTIVE_MANIFOLD_BDF2_BOUNDED_RETRY_PRODUCTION_V1` is default-off and changes
neither the integrator nor any physical/tolerance value. The legacy default remains
`ZERO_REJECT_FIXED_STEP_V1`.

Hard gates are frozen at: zero macro hard rejects; all accepted state gates pass;
reject and fallback fractions <=1%; measured reject-trial wall fraction <=5%;
maximum consecutive fallback macros <=2; accepted subcycle depth <=2; no repeated
worst interface cell on three or more rejected macros; p99 nonlinear iterations
<80% of the 500-iteration budget; and equal-time Ctot/phi/h errors <=2%, profile
<=3%, interface <=0.25 dx with unchanged direction.

Implementation map:

- `pf_params.h:410`: versioned selector storage.
- `bounded_retry_bdf2_utils.h:1-79`: pure host gate and failure taxonomy.
- `main_cuda.cu:31327-31491`: event/accepted-state rollback hashes.
- `main_cuda.cu:34255-34660`: first transport failure freeze.
- `main_cuda.cu:42623-42642`: measured runtime summary.

The contract is an acceptance policy over the existing active-manifold BDF2/event
method. It does not change equations, tolerances, iteration budgets, fallback
mathematics, clipping, or projection.
