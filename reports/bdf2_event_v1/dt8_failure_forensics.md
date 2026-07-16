# dt/8 failure forensics

- The BDF2 preflight finds one infeasible context cell at step 2958.
- Cell 266 has `q_anchor_E=-2.22927232229608308e-11`.
- The selector transactionally uses same-dt Lie-BE; no BDF2 transport trial is
  committed.
- The BE target has zero lower/upper feasibility violations and remains finite.
- The nonlinear solve exhausts its accepted descent path with residual
  `5.13512160604277182e-12` after `214`
  nonlinear iterations and `2884` recorded line-search trials.
- Mass error is `5.68434188608080149e-14` and the accepted
  state is restored.

The root cause is a nonsmooth `q_alpha=0` capacity transition combined with a
same-macro-dt BE fallback.  Increasing the iteration budget is not a production
event treatment because it does not remove the inadmissible BDF2 history pair.
