# Handoff start-time sensitivity

The primary full-PSD/true-area comparison uses 6, 12, 24 and 48 h.  Six hours
is explicitly handoff-sensitive; 12–48 h is the preferred physical coarsening
interval.  The registered hourly aggregate supplies a 10 h Sv/M6 anchor, but
P1 does not invent a 10 h true area or discrete-PSD κ for Cases 004/019.

At 303.15 K on the PF-time matrix/true-area branch, the case order is
`002>003>001>004>021>020>019` at 6 h and `002>020>001>021>019>003>004` at 12 h; therefore early handoff relaxation
does change case ranking.  Nevertheless κ rises toward 48 h for every case
from both 6 h (`True`) and 12 h (`True`), so the 12–48 h
mechanism/sign conclusion is not reversed.

Rank and sign checks are evaluated in `handoff_start_time_sensitivity.csv`.
The 6 h interval changes the magnitude of window closure, while the formal
mechanism decision is based on 12–48 h.  Absolute 6 h window lifetime remains
blocked by handoff conditioning; this does not block the conditional P1
prototype.  No conditioned 6 h state was reconstructed or refitted.
