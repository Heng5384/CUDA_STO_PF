# Reject cluster decision

`isolated_bounded_trial_failures=false`: dt/4 contains five consecutive fallback macros and a depth-4 event, while cells 244/266/267 repeat. dt/8 repeats cell 244 twelve times in the frozen window.

`persistent_method_failure=false`: the accepted endpoints remain extremely close to the fine reference, dt/16 and dt/32 traverse the same physical window with zero rejects, and failures also occur in BE fallback contexts. The proven persistence is in the bound-aware transport nonlinear globalization at coarse dt, not in thermodynamics, active-manifold history, or the BDF2 consistency formula.

Classification: `PERSISTENT_COARSE_DT_TRANSPORT_GLOBALIZATION_FAILURE`. This evidence supports improving the bound-aware transport nonlinear solver; it does not justify implementing SDIRK2/TR-BDF2 in this goal.
