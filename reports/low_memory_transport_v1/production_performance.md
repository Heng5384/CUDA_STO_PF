# Production performance

The only complete zero-retry long-window production baseline is V0 dt/16:
`1.51566303014755249e+02` wall seconds for 8000 steps. V5 dt/16 took
`1.48678366184234619e+02` seconds, an apparent wall change of
`-1.90539504697128104e-02`, but required 82
fallback macros. It therefore fails the production throughput gate regardless
of raw wall time. V5 dt/4's short-window speedup is likewise diagnostic only.

No solver version simultaneously achieved a qualified residual gate, zero
fallback, long-window completion at a larger dt, and the required accuracy
holdouts. `selected_solver_name=NONE`.
